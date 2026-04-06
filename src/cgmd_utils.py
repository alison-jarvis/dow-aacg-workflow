# Imports
from openmm.app import PDBFile
from openmm import CustomNonbondedForce, HarmonicBondForce
from openmm import System
from aamd_utils import *
from openmm.unit import dalton, nanometer, kelvin
from openmm.unit import BOLTZMANN_CONSTANT_kB
from openmm.unit import MOLAR_GAS_CONSTANT_R
from openmm.unit import picosecond, femtoseconds, bar
from openmm import LangevinIntegrator
from openmm.app import Simulation, PDBReporter, DCDReporter, StateDataReporter
from openmm import MonteCarloBarostat
from openmm.unit import norm
from sys import stdout

# Helper function, get bead types from topology object
def get_bead_types(topology):
    bead_types = sorted({atom.name for atom in topology.atoms()})
    return bead_types

# Helper function, get dictionary of bead : gamma / a for nonbonded force
def build_bead_params_srel(topology, initial_gamma, initial_a):

    bead_types = get_bead_types(topology)
    bead_type_params = {}

    for bead in bead_types:
        bead_type_params[bead] = [initial_gamma, initial_a]

    return bead_type_params

# Function to create coarse grained system
def create_cg_system(topology, positions, project_name, identifier, general_config, temperature):

    # Create the system
    system = System()

    # Set periodic box vectors (same as AA system)
    inp_path = project_name + "/" + identifier + ".inp"
    pbc_margin = general_config["periodic box margin"]
    box_vectors = parse_packmol_box(inp_path, pbc_margin=pbc_margin)
    system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

    # Add particles to system
    for atom in topology.atoms():
        # UPDATE - add particle requires bead mass. Should be from universal CG mapping. 
        system.addParticle(54*dalton) # Placeholder bead mass right now (mass of 3 water molecules)

    ######## Option 1 - Srel #############
    if general_config["cg forcefield"] == "srel":

        ############# Non-Bonded #############

        # We define an openMM custom non-bonded force for this gaussian potential
        gaussian_force = CustomNonbondedForce("""sqrt(gamma1*gamma2) * exp(-r^2 / (2*(a1*a1 + a2*a2)))""")

        # Add the critical (iterative) parameters for this force
        gaussian_force.addPerParticleParameter("gamma")
        gaussian_force.addPerParticleParameter("a")

        # Extract user defined values of these parameters for the system
        gamma = general_config["default parameter 1"] # unitless
        a = general_config["default parameter 2"] # nm

        # Convert this into a parameter mapping dictionary based on beads in pdb
        bead_param_mapping = build_bead_params_srel(topology, gamma, a)

        # Set this force with a periodic cutoff and cutoff distance
        gaussian_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
        gaussian_force.setCutoffDistance(general_config["cg interaction cutoff"]*nanometer) # cutoff user defined, from config

        # Add all the beads to the non-bonded force
        for i, atom in enumerate(topology.atoms()):
            bead_type = atom.name
            gamma_bead, a_bead = bead_param_mapping[bead_type] # gamma and a specific to bead type
            gaussian_force.addParticle([gamma_bead, a_bead]) # add these to foce

        # Exclude bonded pairs for this force
        bonds = [(bond[0].index, bond[1].index) for bond in topology.bonds()]
        gaussian_force.createExclusionsFromBonds(bonds, 1)

        # Add non-bonded force to system
        system.addForce(gaussian_force)

        ########### Bonded ##############

        # We implement this with a general harmonic bond force
        bond_force = HarmonicBondForce()

        # Extract the user defined starting value of b from the config
        b = general_config["default parameter 3"]*nanometer # nm

        # Calculate the effective spring constant from b
        kBT = MOLAR_GAS_CONSTANT_R * temperature * kelvin # boltzmann constant (function of temperature)
        k = 3*kBT/(b**2)

        # Iterate through all bonds, add bond force specifically for these indices
        for bond in topology.bonds():
            i = bond[0].index
            j = bond[1].index

            # Calculate distance between particles
            delta = positions[i] - positions[j]
            r0 = norm(delta)

            # Add this bond with initial distance to bond force
            bond_force.addBond(i, j, r0, k)

        # Add force to system
        system.addForce(bond_force)

    return system


def run_cg_simulation(topology, positions, system, general_config, project_name, temperature, simulation_type, pressure, iteration_number=0, save_diagnostics=True):
    # Define Langevin Integrator (temperature, friction, timestep)
    integrator = LangevinIntegrator(temperature, general_config["cg friction"]/picosecond, general_config["cg integration timestep"]*femtoseconds)

    # If the system is NPT, set a barostat
    if simulation_type == "NPT":
        barostat_frequency_steps = steps_from_ns(general_config["cg pressure enforcing frequency"], general_config["cg integration timestep"])
        barostat = MonteCarloBarostat(pressure * bar, temperature, barostat_frequency_steps)
        system.addForce(barostat)

    # Define simulation object and set positions
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # Minimize the system energy
    print("Minimizing system")
    simulation.minimizeEnergy()

    # Get necessary config params to calculate step intervals
    timestep_fs = general_config["cg integration timestep"]
    equil_time_ns = general_config["cg equilibration time"]
    prod_time_ns = general_config["cg production time"]
    traj_freq_ns = general_config["cg trajectory log frequency"]

    # Calculate step quantities for equilibration, production, and reporting
    equil_steps = compute_md_steps(equil_time_ns, timestep_fs)
    prod_steps = compute_md_steps(prod_time_ns, timestep_fs)
    report_interval = compute_report_interval(traj_freq_ns, timestep_fs)

    ########## Equilibration #############

    # Option 1 - NVT
    if simulation_type == "NVT":
        # For NVT, only run at equilibrium at temperature
        print("Equilibrating system")
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(equil_steps)
    # Option 2 - NPT
    elif simulation_type == "NPT":
        # For NPT, first do a small NVT equilibration
        print("Short system equilibration with NVT")
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0 * bar) # temporarily set barostat off
        nvt_equil_steps = steps_from_ps(100.0, general_config["cg integration timestep"])
        simulation.step(nvt_equil_steps) # total 100 ps, constant (not in config)

        # Set the barostat back on and perform full NPT equilibration
        print("NPT equilibration and density relaxation")
        simulation.context.setParameter(MonteCarloBarostat.Pressure(), pressure * bar) # set barostat back on
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(equil_steps)

    ############# Production #############

    # Add reporters for production
    simulation_trajectory_path = project_name + '/cg_trajectories/cgmd_trajectory_' + str(iteration_number) + '.dcd'
    simulation.reporters.append(DCDReporter(simulation_trajectory_path, report_interval))
    simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, temperature=True, density=True)) # another constant, always prints out at frequency of 5000 steps

    if save_diagnostics:
        log_path = project_name + '/cg_diagnostics/cg_log_' + str(iteration_number) + '.txt'
        print(f"Reported quantities from CG simulation (T, E, \u03C1, V) will be written to the file: {log_path}")
        simulation.reporters.append(StateDataReporter(log_path, report_interval, step=True, temperature=True, 
                                                      potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                                      density=True, volume=True))

    # Run production
    print("Running production")
    simulation.step(prod_steps)