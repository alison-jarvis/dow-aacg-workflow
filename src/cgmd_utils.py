# Imports
import os
import numpy as np
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from sys import stdout
import sympy as sp
import re
from openmm.unit import *
from openmm.app import Simulation, DCDReporter, StateDataReporter
from openmm import CustomNonbondedForce, Discrete2DFunction, CustomBondForce
from openmm import System, MonteCarloBarostat, LangevinIntegrator
from aamd_utils import *

############### Parameter Handling Utils ##################

# Pair key type
PairKey = Tuple[str, str]

# Defines the unique bead ID for an atom
def bead_instance_id(atom) -> str:
    return f"{atom.residue.name.strip()}_{atom.name.strip()}"

# Defines the unique bead type ID for an atom
def bead_type_id(atom) -> str:
    return atom.residue.name.strip()

# Enforces ordering of bead ID pairs for consistency
def canonical_pair(a: str, b: str) -> PairKey:
    return tuple(sorted((a, b)))

# Get set of unique bead types for a topology
def get_unique_bead_types(topology):
    return sorted({bead_type_id(atom) for atom in topology.atoms()})

# Get set of unique pairs for a topology
def get_all_type_pairs(topology):
    bead_types = get_unique_bead_types(topology)
    return [canonical_pair(a, b) for a, b in itertools.combinations_with_replacement(bead_types, 2)]

# Get set of unique bonded pairs for a topology
def get_bonded_type_pairs(topology):
    bonded_pairs = set()
    for bond in topology.bonds():
        bt_i = bead_type_id(bond[0])
        bt_j = bead_type_id(bond[1])
        bonded_pairs.add(canonical_pair(bt_i, bt_j))
    return sorted(bonded_pairs)

# Get mapping of bead ID to bead mass from topology
def extract_bead_mass_mapping(topology):
    """TO DO: IMPLEMENT - NOT CORRECT!!!"""
    bead_masses = dict()
    for atom in topology.atoms():
        unique_id = bead_type_id(atom)
        bead_masses[unique_id] = 18
    return bead_masses

############# Forcefield Handling Utils ##############

# Get parameter names from the config
def extract_parameter_names(parameter_config, section, ptype):
    if section not in parameter_config:
        return []

    return [pname for pname, spec in parameter_config[section].items() if spec["type"] == ptype]


# Convert basic equation into openmm expectation for nonbonded 
def insert_openmm_pair_lookups(expr, pairwise_names):
    """
    Convert symbolic nonbonded expression like:
        gamma*exp(...)
    into openmm ready expression:
        gamma(type1,type2)*exp(...)
    specifically adding type reference to all pairwise parameters
    """
    out = expr
    for pname in pairwise_names:
        out = re.sub(rf"\b{re.escape(pname)}\b", f"{pname}(type1,type2)", out)
    return out

# Create dictionary of local symbols for sympy
def build_sympy_locals(var_names):
    """
    Build locals dictionary for sympify
    Overwrites any potential built-in functions with these variable names
    """
    return {name: sp.Symbol(name) for name in var_names}

# Build the first and second derivative strings with sympy
def build_derivative_functions(expr_string, eval_param_names, differentiate_names):
    """
    Build first and second derivative dictionaries from an expression string

    Returns
    -------
    first_exprs, first_funcs (first derivative string and lambda function)
    second_exprs, second_funcs (second derivative string and lambda function)
    """
    # Construct the custom local symbols from parameter names
    custom_locals = build_sympy_locals(eval_param_names)
    # Parse the equation string into a sympy expression
    parsed_expr = sp.sympify(expr_string, locals=custom_locals)

    # Make a list of sympy symbols to differentiate wrt
    eval_symbols = tuple(custom_locals[name] for name in eval_param_names)

    # Initialize expressions and functions
    first_exprs, first_funcs, second_exprs, second_funcs = {}, {}, {}, {}

    # Iterate through parameters for PDEs
    for pname in differentiate_names:
        sym = custom_locals[pname]

        # Get the first derivative as string
        first_deriv = sp.diff(parsed_expr, sym)
        first_exprs[pname] = first_deriv
        # Convert first derivative to sympy lambda function (with numpy! important so its fast)
        first_funcs[pname] = sp.lambdify(eval_symbols, first_deriv, modules="numpy")

        # Get the second derivative as string
        second_deriv = sp.diff(first_deriv, sym)
        second_exprs[pname] = second_deriv
        # Convert second derivative to sympy lambda function
        second_funcs[pname] = sp.lambdify(eval_symbols, second_deriv, modules="numpy")

    return first_exprs, first_funcs, second_exprs, second_funcs


################# Parameter Object ##################

@dataclass
class ParameterSet:
    pair_names: List[str]
    individual_names: List[str]
    fixed_names: List[str]

    # Pair parameters - dictionary of parameter name : bead pair, default value
    pair_parameters: Dict[str, Dict[PairKey, float]] = field(default_factory=dict)

    # Individual parameters - dictionary of parameter name : bead name, default value
    individual_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Fixed parameters - dictionary of parameter name : default value
    fixed_parameters: Dict[str, object] = field(default_factory=dict)

    # Flags - whether to update each parameter according to config
    pair_update_flags: Dict[str, bool] = field(default_factory=dict)
    individual_update_flags: Dict[str, bool] = field(default_factory=dict)
    fixed_update_flags: Dict[str, bool] = field(default_factory=dict)

    ########### Functions for ease of class use ############
    def get_pair(self, pair: PairKey, name: str) -> float:
        return self.pair_parameters[name][canonical_pair(*pair)]

    def set_pair(self, pair: PairKey, name: str, value: float) -> None:
        self.pair_parameters[name][canonical_pair(*pair)] = value

    def has_pair(self, pair: PairKey, name: str) -> bool:
        return canonical_pair(*pair) in self.pair_parameters.get(name, {})

    def get_individual(self, bead_type: str, name: str) -> float:
        return self.individual_parameters[name][bead_type]

    def set_individual(self, bead_type: str, name: str, value: float) -> None:
        self.individual_parameters[name][bead_type] = value

    def has_individual(self, bead_type: str, name: str) -> bool:
        return bead_type in self.individual_parameters.get(name, {})

    def get_fixed(self, name: str):
        return self.fixed_parameters[name]

    def set_fixed(self, name: str, value) -> None:
        self.fixed_parameters[name] = value

    # Update parameter object from calculated gradient values
    def apply_gradients(self, gradients, learning_rate):
        """
        Parameters
        ----------
        gradients : dict
            Dictionary of parameter type : {name : gradient value}
        learning_rate : float
            For standard gradient descent, how much to move along the gradient
        """

        # Pairwise parameters
        for pname, param_dict in gradients["pair"].items():
            for key, grad in param_dict.items():
                # Check whether to update this parameter
                if not self.pair_update_flags.get(pname, True):
                    continue
                # Update the value, enforcing > 0
                value = self.pair_parameters[pname][key] - learning_rate * grad
                self.pair_parameters[pname][key] = max(value, 1e-6)

        # Individual parameters
        for pname, param_dict in gradients["individual"].items():
            for key, grad in param_dict.items():
                # Check whether to update this parameter
                if not self.individual_update_flags.get(pname, True):
                    continue
                # Update the value, enforcing > 0
                value = self.individual_parameters[pname][key] - learning_rate * grad
                self.individual_parameters[pname][key] = max(value, 1e-6)

        # Fixed parameters
        for pname, grad in gradients["fixed"].items():
            # Check whether to update this parameter
            if not self.fixed_update_flags.get(pname, True):
                continue
            # Update the value, enforcing > 0
            value = self.fixed_parameters[pname] - learning_rate * grad
            self.fixed_parameters[pname] = max(value, 1e-6)

############## ForceField Object ##################

@dataclass
class ForceSpec:

    # Define the expressions themselves
    nonbonded_expression: Optional[str] = None
    nonbonded_expression_openmm: Optional[str] = None
    bonded_expression: Optional[str] = None

    # Define the names of the parameters from expressions
    nonbonded_pair_names: List[str] = field(default_factory=list)
    nonbonded_individual_names: List[str] = field(default_factory=list)
    nonbonded_fixed_names: List[str] = field(default_factory=list)

    bonded_pair_names: List[str] = field(default_factory=list)
    bonded_individual_names: List[str] = field(default_factory=list)
    bonded_fixed_names: List[str] = field(default_factory=list)

    # Ordered variable lists used for lambdify, needed to evaluate
    nonbonded_eval_params: List[str] = field(default_factory=list)
    bonded_eval_params: List[str] = field(default_factory=list)

    # First derivative callable lambda functions
    nonbonded_first_derivatives: Dict[str, Callable] = field(default_factory=dict)
    bonded_first_derivatives: Dict[str, Callable] = field(default_factory=dict)

    # Second derivative callable lambda functions (same parameters as first derivatives)
    nonbonded_second_derivatives: Dict[str, Callable] = field(default_factory=dict)
    bonded_second_derivatives: Dict[str, Callable] = field(default_factory=dict)

    # Save symbolic expressions for debugging / transparency
    nonbonded_first_derivative_exprs: Dict[str, sp.Expr] = field(default_factory=dict)
    bonded_first_derivative_exprs: Dict[str, sp.Expr] = field(default_factory=dict)

    nonbonded_second_derivative_exprs: Dict[str, sp.Expr] = field(default_factory=dict)
    bonded_second_derivative_exprs: Dict[str, sp.Expr] = field(default_factory=dict)


############# Config Parsing Utils #################

# Parse parameter from the config file
def parse_parameters_from_config(parameter_config, topology, temperature):
    """
    Load parameters from config file into a Parameter object
    Parameter Categories:
    - Nonbonded
        - Pairwise
        - Individual
        - Fixed
    - Bonded
        - Pairwise
        - Individual
        - Fixed
    """

    # Initialize objects to fill from config
    pair_names = []
    individual_names = []
    fixed_names = []

    pair_update_flags = {}
    individual_update_flags = {}
    fixed_update_flags = {}

    pair_parameters = {}
    individual_parameters = {}
    fixed_parameters = {}

    # Get useful mappings from topology
    all_bead_types = get_unique_bead_types(topology)
    all_type_pairs = get_all_type_pairs(topology)
    bonded_type_pairs = get_bonded_type_pairs(topology)

    ############ Non-bonded Parameters #############
    if "nonbonded" in parameter_config:
        for pname, spec in parameter_config["nonbonded"].items():
            ptype = spec["type"]
            default = spec["default"]
            update_flag = spec.get("update", True)

            if ptype == "pairwise":
                pair_names.append(pname)
                pair_parameters[pname] = {pair: float(default) for pair in all_type_pairs}
                pair_update_flags[pname] = update_flag

            elif ptype == "individual":
                individual_names.append(pname)
                individual_parameters[pname] = {bead_type: float(default) for bead_type in all_bead_types}
                individual_update_flags[pname] = update_flag

            elif ptype == "fixed":
                fixed_names.append(pname)
                # Accounts for potential inclusion of Boltzmann constant as fixed param
                if pname == "kBT":
                    kBT = MOLAR_GAS_CONSTANT_R * temperature * kelvin
                    fixed_parameters[pname] = kBT
                else:
                    fixed_parameters[pname] = default
                fixed_update_flags[pname] = update_flag

            else:
                raise ValueError(f"Unknown parameter type '{ptype}' for '{pname}'")

    ########### Bonded Parameters #############
    if "bonded" in parameter_config:
        for pname, spec in parameter_config["bonded"].items():
            ptype = spec["type"]
            default = spec["default"]
            update_flag = spec.get("update", True)

            if ptype == "pairwise":
                if pname not in pair_names:
                    pair_names.append(pname)
                pair_parameters[pname] = {pair: float(default) for pair in bonded_type_pairs}
                pair_update_flags[pname] = update_flag

            elif ptype == "individual":
                if pname not in individual_names:
                    individual_names.append(pname)
                individual_parameters[pname] = {bead_type: float(default) for bead_type in all_bead_types}
                individual_update_flags[pname] = update_flag

            elif ptype == "fixed":
                if pname not in fixed_names:
                    fixed_names.append(pname)
                # Accounts for potential inclusion of Boltzmann constant as fixed param
                if pname == "kBT":
                    kBT = MOLAR_GAS_CONSTANT_R * temperature * kelvin
                    fixed_parameters[pname] = kBT
                else:
                    fixed_parameters[pname] = default
                fixed_update_flags[pname] = update_flag


            else:
                raise ValueError(f"Unknown parameter type '{ptype}' for '{pname}'")
            

    # Define parameter set object from all of the extracted quantities
    ps = ParameterSet(pair_names=pair_names, individual_names=individual_names, 
                      fixed_names=fixed_names, pair_parameters=pair_parameters,
                      individual_parameters=individual_parameters, fixed_parameters=fixed_parameters,
                      pair_update_flags=pair_update_flags, individual_update_flags=individual_update_flags,
                      fixed_update_flags=fixed_update_flags)
    return ps

# Parse forcefield parameters from config
def parse_forcefield_from_config(general_config, parameter_config):
    """
    Parse parameters corresponding to the forcefield config

    Parameters
    ----------
    general_config : dict
        Needs to contain:
            - "nonbonded expression"
            - "bonded expression"
    parameter_config : dict
        Defines characteristics of the parameters contained within the expressions

    Returns
    -------
    ForceSpec
    """
    # Extract the bonded and nonbonded expressions
    nonbonded_expression = general_config.get("nonbonded expression", None)
    bonded_expression = general_config.get("bonded expression", None)

    # Get parameter names for each category
    nonbonded_pair_names = extract_parameter_names(parameter_config, "nonbonded", "pairwise")
    nonbonded_individual_names = extract_parameter_names(parameter_config, "nonbonded", "individual")
    nonbonded_fixed_names = extract_parameter_names(parameter_config, "nonbonded", "fixed")

    bonded_pair_names = extract_parameter_names(parameter_config, "bonded", "pairwise")
    bonded_individual_names = extract_parameter_names(parameter_config, "bonded", "individual")
    bonded_fixed_names = extract_parameter_names(parameter_config, "bonded", "fixed")

    # Automatically includes kBT as bonded/nonbonded fixed if the expression uses it
    # (if user didn't add kBT in the parameters)
    if bonded_expression is not None and re.search(r"\bkBT\b", bonded_expression):
        if "kBT" not in bonded_fixed_names:
            bonded_fixed_names.append("kBT")

    if nonbonded_expression is not None and re.search(r"\bkBT\b", nonbonded_expression):
        if "kBT" not in nonbonded_fixed_names:
            nonbonded_fixed_names.append("kBT")

    # Create the openmm versions of the nonbonded expression (with types)
    nonbonded_expression_openmm = None
    if nonbonded_expression is not None:
        nonbonded_expression_openmm = insert_openmm_pair_lookups(nonbonded_expression, nonbonded_pair_names)

    ###### Symbolic Differentiation ######

    # Nonbonded expression variables - start with radii, update for angles and torsion
    nonbonded_eval_params = ["r"]

    # Add the pairwise parameters in symbolic form
    nonbonded_eval_params.extend(nonbonded_pair_names)

    # Add individual parameters as pname1, pname2 (required for openmm individual)
    nonbonded_diff_names = list(nonbonded_pair_names)
    for pname in nonbonded_individual_names:
        nonbonded_eval_params.extend([f"{pname}1", f"{pname}2"])
        nonbonded_diff_names.extend([f"{pname}1", f"{pname}2"])

    # Add fixed names
    nonbonded_eval_params.extend(nonbonded_fixed_names)
    nonbonded_diff_names.extend(nonbonded_fixed_names)

    # Housekeeping, removes any duplicates
    nonbonded_eval_params = list(dict.fromkeys(nonbonded_eval_params))
    nonbonded_diff_names = list(dict.fromkeys(nonbonded_diff_names))

    # Bonded expression variables - again only for radii so far
    bonded_eval_params = ["r"]
    bonded_eval_params.extend(bonded_pair_names)

    # Individual parameters (pname1, pname2)
    bonded_diff_names = list(bonded_pair_names)
    for pname in bonded_individual_names:
        bonded_eval_params.extend([f"{pname}1", f"{pname}2"])
        bonded_diff_names.extend([f"{pname}1", f"{pname}2"])

    # Add fixed parameters
    bonded_eval_params.extend(bonded_fixed_names)
    bonded_diff_names.extend(bonded_fixed_names)

    # Housekeeping
    bonded_eval_params = list(dict.fromkeys(bonded_eval_params))
    bonded_diff_names = list(dict.fromkeys(bonded_diff_names))

    # Build the derivative dictionaries (nonbonded)
    nb_first_exprs, nb_first_funcs, nb_second_exprs, nb_second_funcs = {}, {}, {}, {}

    if nonbonded_expression is not None:
        (nb_first_exprs, nb_first_funcs, nb_second_exprs, nb_second_funcs) = \
        build_derivative_functions(nonbonded_expression, nonbonded_eval_params, 
                                   nonbonded_diff_names)

    # Build the derivative dictionaries (bonded)
    bd_first_exprs, bd_first_funcs, bd_second_exprs, bd_second_funcs = {}, {}, {}, {}

    if bonded_expression is not None:
        (bd_first_exprs, bd_first_funcs, bd_second_exprs, bd_second_funcs) = \
            build_derivative_functions(bonded_expression, bonded_eval_params, 
                                       bonded_diff_names)

    # Build the actual force object
    ff = ForceSpec(

        nonbonded_expression=nonbonded_expression,
        nonbonded_expression_openmm=nonbonded_expression_openmm,
        bonded_expression=bonded_expression,

        nonbonded_pair_names=nonbonded_pair_names,
        nonbonded_individual_names=nonbonded_individual_names,
        nonbonded_fixed_names=nonbonded_fixed_names,

        bonded_pair_names=bonded_pair_names,
        bonded_individual_names=bonded_individual_names,
        bonded_fixed_names=bonded_fixed_names,

        nonbonded_eval_params=nonbonded_eval_params,
        bonded_eval_params=bonded_eval_params,

        nonbonded_first_derivatives=nb_first_funcs,
        bonded_first_derivatives=bd_first_funcs,

        nonbonded_second_derivatives=nb_second_funcs,
        bonded_second_derivatives=bd_second_funcs,

        nonbonded_first_derivative_exprs=nb_first_exprs,
        bonded_first_derivative_exprs=bd_first_exprs,

        nonbonded_second_derivative_exprs=nb_second_exprs,
        bonded_second_derivative_exprs=bd_second_exprs,
    )

    return ff

############### ForceField Utils ################

# Build a custom nonbonded force from parameters and forcefield
def build_nonbonded_force(topology, parameter_set: ParameterSet, force_spec: ForceSpec, cutoff_nm: float):
    """Custom nonbonded openmm force object"""

    # Parse unique bead types from either parameters or topology
    bead_types = sorted(next(iter(parameter_set.individual_parameters.values())).keys()) \
        if parameter_set.individual_parameters else get_unique_bead_types(topology)

    # Map the bead indices to types
    type_index_map = {bt: i for i, bt in enumerate(bead_types)}
    n_types = len(bead_types)

    # Define a custom nonbonded force with the expression
    force = CustomNonbondedForce(force_spec.nonbonded_expression_openmm)

    # Add type as per-particle parameter, so type is indexed by i, j
    force.addPerParticleParameter("type")

    # Add individual parameters as per particle parameters
    for name in force_spec.nonbonded_individual_names:
        force.addPerParticleParameter(name)

    # Add fixed parameters, which are globals (not bead specific)
    for name in force_spec.nonbonded_fixed_names:
        force.addGlobalParameter(name, parameter_set.get_fixed(name))

    # For each nonbonded parameter name, initialize table of bead type pairs
    for pname in force_spec.nonbonded_pair_names:
        table = np.zeros((n_types, n_types), dtype=float)
        for ti, bt_i in enumerate(bead_types):
            for tj, bt_j in enumerate(bead_types):
                table[ti, tj] = parameter_set.get_pair((bt_i, bt_j), pname)

        # Add tabulated function representing pair-parameters to force
        pname_as_function = Discrete2DFunction(n_types, n_types, table.flatten().tolist())
        force.addTabulatedFunction(pname, pname_as_function)

    # Add particles to force with default values of individual parameters
    for atom in topology.atoms():
        bt = bead_type_id(atom)
        values = [float(type_index_map[bt])]
        for name in force_spec.nonbonded_individual_names:
            values.append(parameter_set.get_individual(bt, name))
        force.addParticle(values)

    # Set periodic cutoff for this force
    force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff_nm * nanometer)

    # Exclude bonded particles from nonbonded force
    bonds = [(bond[0].index, bond[1].index) for bond in topology.bonds()]
    force.createExclusionsFromBonds(bonds, 1)

    return force


# Build a custom openmm bonded force objcet
def build_bonded_force(topology, parameter_set: ParameterSet, force_spec: ForceSpec):
    """Custom bonded force from parameters and forcefield"""

    # Create custom bonded force from expression
    force = CustomBondForce(force_spec.bonded_expression)

    # Add per-bond parameter names to force (pair names)
    for name in force_spec.bonded_pair_names:
        force.addPerBondParameter(name)

    # Add per bead parameters to force (individual names), called 1 and 2
    for name in force_spec.bonded_individual_names:
        force.addPerBondParameter(f"{name}1")
        force.addPerBondParameter(f"{name}2")

    # Add fixed global parameters to force
    for name in force_spec.bonded_fixed_names:
        force.addGlobalParameter(name, parameter_set.get_fixed(name))

    # Iterate through bonds
    for bond in topology.bonds():
        # Indices of bond pair
        i = bond[0].index
        j = bond[1].index

        # Get i / j bead IDs and pair ID
        bt_i = bead_type_id(bond[0])
        bt_j = bead_type_id(bond[1])
        pair = canonical_pair(bt_i, bt_j)

        # Net values to add to bond
        values = []

        # Append pairwise parameters to values for this bond
        for name in force_spec.bonded_pair_names:
            if not parameter_set.has_pair(pair, name):
                raise KeyError(f"Missing bonded pair parameter '{name}' for pair {pair}")
            values.append(parameter_set.get_pair(pair, name))

        # Append individual parameters to values for this bond
        for name in force_spec.bonded_individual_names:
            if not parameter_set.has_individual(bt_i, name):
                raise KeyError(f"Missing individual bonded parameter '{name}' for bead type {bt_i}")
            if not parameter_set.has_individual(bt_j, name):
                raise KeyError(f"Missing individual bonded parameter '{name}' for bead type {bt_j}")
            values.append(parameter_set.get_individual(bt_i, name))
            values.append(parameter_set.get_individual(bt_j, name))

        # Add this bond to the force
        force.addBond(i, j, values)

    return force

# Build a generalized coarse grained system from parameter / force
def build_general_cg_system(topology, project_name, identifier, masses, parameter_set: ParameterSet,
                            force_spec: ForceSpec, general_config: Dict):
    
    # Define the cutoff from the general config file
    cutoff_nm = general_config["cg interaction cutoff"]

    # Initialize a system
    system = System()

    # Set periodic box vectors (same as AA system)
    inp_path = project_name + "/" + identifier + ".inp"
    pbc_margin = general_config["periodic box margin"]
    box_vectors = parse_packmol_box(inp_path, pbc_margin=pbc_margin)
    system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

    # Iterate through atoms in topology and add the particle to the system
    for atom in topology.atoms():
        bt = bead_type_id(atom)
        system.addParticle(masses[bt] * dalton)

    # Add the nonbonded force to the system
    if force_spec.nonbonded_expression is not None:
        nonbonded_force = build_nonbonded_force(topology, parameter_set, force_spec, cutoff_nm)
        system.addForce(nonbonded_force)

    # Add the bonded force to the system
    if force_spec.bonded_expression is not None:
        bonded_force = build_bonded_force(topology, parameter_set, force_spec)
        system.addForce(bonded_force)

    return system


################ CG Simulation Utils ################

# Function to run a simulation
def run_cg_simulation(topology, positions, system, general_config, project_name, temperature, 
                      simulation_type, pressure, iteration_number=0, save_diagnostics=True):
    
    # Folder management, make sure trajectories and diagnostics exist
    traj_dir = f"./{project_name}/cg_trajectories/"
    os.makedirs(traj_dir, exist_ok=True)
    if save_diagnostics:
        diag_dir = f"./{project_name}/cg_diagnostics/"
        os.makedirs(diag_dir, exist_ok=True)

    # Define a langevin integrator with temperature, friction, and timestep
    integrator = LangevinIntegrator(temperature, 
        general_config["cg friction"]/picosecond, 
        general_config["cg integration timestep"]*femtoseconds)

    # Special case for NPT simulation, add barostat
    if simulation_type == "NPT":
        barostat_freq = steps_from_ns(general_config["cg pressure enforcing frequency"], general_config["cg integration timestep"])
        barostat = MonteCarloBarostat(pressure * bar, temperature, barostat_freq)
        system.addForce(barostat)

    # Define a simulation object, set positions
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # Minimize the energy of the system
    print(f"Minimizing system for iteration {iteration_number}...")
    simulation.minimizeEnergy()

    # Calculate step / reporting quantities from config
    timestep_fs = general_config["cg integration timestep"]
    equil_steps = compute_md_steps(general_config["cg equilibration time"], timestep_fs)
    prod_steps = compute_md_steps(general_config["cg production time"], timestep_fs)
    report_interval = compute_report_interval(general_config["cg trajectory log frequency"], timestep_fs)

    # Equilibrate the system
    if simulation_type == "NVT":
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(equil_steps)
    elif simulation_type == "NPT":
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0 * bar) 
        simulation.step(steps_from_ps(100.0, timestep_fs))
        simulation.context.setParameter(MonteCarloBarostat.Pressure(), pressure * bar) 
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(equil_steps)

    # Set production reporters
    traj_path = f"{traj_dir}/cgmd_trajectory_{iteration_number}.dcd"
    simulation.reporters.append(DCDReporter(traj_path, report_interval))
    simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, temperature=True, density=True))

    # Define reporter if saving diagnostics
    if save_diagnostics:
        log_path = f"{diag_dir}/cg_log_{iteration_number}.txt"
        simulation.reporters.append(StateDataReporter(log_path, report_interval, step=True,
                                                       temperature=True, potentialEnergy=True, 
                                                       kineticEnergy=True, totalEnergy=True, 
                                                       density=True, volume=True))

    # Run production
    print(f"Running production for iteration {iteration_number}...")
    simulation.step(prod_steps)

    # Return the final positions
    return simulation.context.getState(getPositions=True).getPositions()
