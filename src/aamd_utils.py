from openff.toolkit import Molecule, Topology
import re
import numpy as np
from openmm.app import ForceField
from openmm.unit import nanometer
from openmm.app import PME, HBonds
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit.topology import Molecule
from openmm.app import PDBFile
from openff.units.openmm import from_openmm
from openff.units import unit as off_unit
from openmm.unit import picosecond, femtoseconds, bar
from openmm import LangevinIntegrator
from openmm.app import Simulation, PDBReporter, StateDataReporter
from openmm import MonteCarloBarostat
from sys import stdout
import pandas as pd
import os
import matplotlib.pyplot as plt
import mdtraj as md
import nglview as nv
from pathlib import Path

################### Utility Functions #####################

def compute_md_steps(time_ns, timestep_fs):
    """
    Convert time in ns and timestep in fs to integer number of MD steps
    """
    steps = int((time_ns * 1e6) / timestep_fs)
    return steps

def compute_report_interval(freq_ns, timestep_fs):
    """
    Convert output frequency (ns) to reporter step interval
    """
    interval = int((freq_ns * 1e6) / timestep_fs)
    return max(1, interval)

def steps_from_ps(time_ps, timestep_fs):
    """
    Convert general time in picoseconds to number of steps
    """
    time_fs = time_ps * 1000.0
    return int(time_fs / timestep_fs)

def steps_from_ns(time_ns, timestep_fs):
    """
    Convert general time in nanoseconds to number of steps
    """
    time_fs = time_ns * 1e6
    return int(time_fs / timestep_fs)

def parse_general_config(config_path):
    """
    Parse information from the general config file, return as dictionary
    """
    config_dict = {}

    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            left, value = line.split(":", 1)
            key = left.split(",")[0].strip()
            value = value.strip()

            # Convert numeric values automatically
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string

            if value == 'None' or value == 'none':
                value = None

            config_dict[key] = value

    return config_dict

def plot_diagnostics(log_path, project_name):
    # Check if a plots folder exists in the project
    plot_folder = project_name + '/plots/'
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder) # if not, make one

    # Load in the log file as a pandas df
    df = pd.read_csv(log_path)

    # Create the plots

    ########## Energies ###########
    fig, ax = plt.subplots(3, sharex=True, figsize=(8, 6))
    ax[0].plot(df['#"Step"'], df['Potential Energy (kJ/mole)'])
    ax[0].set_title('Potential Energy', loc='right')
    ax[0].set_ylabel('Energy (kJ/mol)')
    ax[1].plot(df['#"Step"'], df['Kinetic Energy (kJ/mole)'])
    ax[1].set_title('Kinetic Energy', loc='right')
    ax[1].set_ylabel('Energy (kJ/mol)')
    ax[2].plot(df['#"Step"'], df['Total Energy (kJ/mole)'])
    ax[2].set_title('Total Energy', loc='right')
    ax[2].set_ylabel('Energy (kJ/mol)')
    ax[2].set_xlabel('Simulation Step')
    plt.suptitle('Simulation Energies')
    fig.tight_layout()
    fig.savefig(plot_folder + 'energy.png')

    ########### Temperature #############
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df['#"Step"'], df["Temperature (K)"])
    plt.suptitle('Simulation Temperature')
    ax.set_ylabel('Temperature (K)')
    ax.set_xlabel('Simulation Step')
    fig.tight_layout()
    fig.savefig(plot_folder + 'temperature.png')

    ########### Density #############
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(df['#"Step"'], df["Density (g/mL)"])
    plt.suptitle('Simulation Density')
    ax.set_ylabel('Density (g/mL)')
    ax.set_xlabel('Simulation Step')
    fig.tight_layout()
    fig.savefig(plot_folder + 'density.png')

    ########### Volume #############
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df['#"Step"'], df["Box Volume (nm^3)"])
    plt.suptitle('Simulation Volume')
    ax.set_ylabel('Volume ($nm^3$)')
    ax.set_xlabel('Simulation Step')
    fig.tight_layout()
    fig.savefig(plot_folder + 'volume.png')

def visualize_trajectory(project_name):
    # Find a trajectory file for this project
    traj_files = list(Path(project_name).glob('*.dcd'))
    if not traj_files:
        raise Exception(f"No trajectory files exist for the project {project_name}.")
    traj_file = str(traj_files[0])
    # We assume for now if the trajectory file exists, a corresponding pdb has to exist (was used to create)
    identifier = re.search(r'/(.*?)\.dcd', traj_file).group(1)
    pdb_file = project_name + '/' + identifier + '.pdb'

    traj = md.load(traj_file, top=pdb_file)
    view = nv.show_mdtraj(traj)
    return view

##################### AAMD Functions ###################

def generate_topology(config_path, identifier, project_name, pbc_margin=0):
    """
    Get a topology object from the config and packed box
    """
    # Create topology object from the config file
    topology, molecules = create_topology_from_config(config_path)

    # Load in the full system pdb file and extract positions
    pdb_path = project_name + "/" + identifier + ".pdb"
    pdb = PDBFile(pdb_path)
    positions = pdb.positions

    # Set topology positions to pdb positions
    positions_off_units = from_openmm(positions) # converts positions to openff units
    topology.set_positions(positions_off_units)

    # Load in box information from pdb input file
    inp_path = project_name + "/" + identifier + ".inp"
    box_vectors = parse_packmol_box(inp_path, pbc_margin=pbc_margin)

    # Set topology box vectors for periodic boundary condition
    topology.box_vectors = off_unit.Quantity(box_vectors, off_unit.nanometer)

    # Convert molecule dict to list of molecule objects
    molecule_list = list(molecules.values())

    return topology, molecule_list, positions


def parse_simulation_parameters(config_path):
    """
    Reads in the simulation config file and pulls out overall system parameters

    Parameters
    ----------
    config_path : str
        Path to the system configuration file

    Returns
    -------
    project_name : str
        Name of the project (folder outputs are in)
    temperature : float
        Temperature of the system (in Kelvin)
    pressure : float
        Pressure of the system (in bars)
    simulation_type : str
        Type of the simulation, options NPT or NVT
    """

    # Define defaults of None
    temperature, pressure, simulation_type = None, None, None

    # Read the config file
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("project name"):
                project_name = line.split(":")[1].strip()

            # Extract temperature (assumes Kelvin)
            elif line.startswith("temperature"):
                temperature = float(line.split(":")[1].strip())

            # Extracts pressure (assumes bars)
            elif line.startswith("pressure"):
                pressure = float(line.split(":")[1].strip())

            # Extracts simulation type
            elif line.startswith("simulation type"):
                simulation_type = line.split(":")[1].strip()

    if temperature is None:
        raise ValueError("Temperature not found in config file.")
    if simulation_type is None:
        raise ValueError("Simulation type not found in config file.")
    
    return project_name, pressure, temperature, simulation_type


def parse_packmol_box(inp_path, pbc_margin=0.0, units="angstrom"):
    """
    Parse the packmol input file to get periodic box dimensions

    Parameters
    ----------
    inp_path : str
        Path to the system input file (.inp)
    pbc_margin : float
        Percentage to expand box margins (equal in all directions)
    units : str
        angstrom (default) or nanometer

    Returns
    -------
    box_lengths : tuple
        (Lx, Ly, Lz) in nanometers
    box_vectors : np.ndarray
        3x3 box vectors in nanometers
    """

    # Initialize list of inside box variables from inp
    inside_boxes = []

    # Read in input file and populate those
    with open(inp_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("inside box"):
                # Extract the six numbers with regex pattern
                numbers = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                if len(numbers) == 6:
                    inside_boxes.append(numbers)

    # Convert to array
    inside_boxes = np.array(inside_boxes)

    # Get overall global bounds from all individual box bounds
    xlo = np.min(inside_boxes[:, 0])
    ylo = np.min(inside_boxes[:, 1])
    zlo = np.min(inside_boxes[:, 2])

    xhi = np.max(inside_boxes[:, 3])
    yhi = np.max(inside_boxes[:, 4])
    zhi = np.max(inside_boxes[:, 5])

    # Computer overall box lengths
    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo

    # Apply margin (percentage expansion)
    scale_factor = 1.0 + pbc_margin / 100.0

    Lx *= scale_factor
    Ly *= scale_factor
    Lz *= scale_factor

    # Potentially convert units from nm to angstrom (packmol angstrom by default)
    if units == "angstrom":
        Lx *= 0.1
        Ly *= 0.1
        Lz *= 0.1

    # Create array of box vectors for topology object
    box_vectors = np.array([[Lx, 0.0, 0.0],
                            [0.0, Ly, 0.0],
                            [0.0, 0.0, Lz]])

    return box_vectors


def create_topology_from_config(config_path):
    """
    Read a config file and construct an openff topology object

    Parameters
    ----------
    config_path : str
        Path to config file

    Returns
    -------
    topology : Topology object
    molecule_dict : dict of smiles: Molecule object
    """

    # Lists for solvents and their numbers
    solvents = []
    solvent_counts = []

    # Lists for compounds of interest and their numbers
    coi_smiles = []
    coi_counts = []

    # Read important info from the config file
    with open(config_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:

        if line.startswith("solvents:"):
            solvents = line.split(":")[1].split()

        elif line.startswith("number of solvent molecules:"):
            solvent_counts = [int(x) for x in line.split(":")[1].split()]

        elif line.startswith("compounds of interest:"):
            coi_smiles = line.split(":")[1].split()

        elif line.startswith("number of CoI molecules:"):
            coi_counts = [int(x) for x in line.split(":")[1].split()]

    # Make sure lists of solvents and their counts match
    if len(solvents) != len(solvent_counts):
        raise ValueError("Mismatch between solvent SMILES and solvent counts")

    if len(coi_smiles) != len(coi_counts):
        raise ValueError("Mismatch between CoI SMILES and CoI counts")

    # Create the molecule objects
    molecule_dict = dict()

    # Add together solvent and COI lists, generate molecules and conformers for all
    for smi in solvents + coi_smiles:
        if smi not in molecule_dict:
            mol = Molecule.from_smiles(smi)
            mol.generate_conformers(n_conformers=1)
            molecule_dict[smi] = mol

    # Get molecules into list form to create topologies
    topology_molecules = []

    for smi, count in zip(solvents, solvent_counts):
        topology_molecules.extend([molecule_dict[smi]] * count)

    for smi, count in zip(coi_smiles, coi_counts):
        topology_molecules.extend([molecule_dict[smi]] * count)

    # Create the topology object from molecule list
    topology = Topology.from_molecules(topology_molecules)

    return topology, molecule_dict


# Function to create system from any inputs
def create_universal_system(topology, molecules, general_config):
    """
    Takes openff topology object and forcefield parameters to create sysetm

    Parameters
    ----------
    topology : openff topology object
        Contains molecules in system and corresponding locations from pdb
    molecules : list of openff Molecule objects 
        (needed for GAFF version of amber)
    ff_type : str
        Forcefield type, either "openff" or "amber" currently
    water_model : str
        Whether to use an explicit water model, options None, "tip3" or "tip4"

    Returns
    -------
    system : openmm system object
        System object for an openmm simulation, corresponding to topology and forcefield

    """

    # Read in the specific general config parameters needed
    ff_type = general_config["forcefield"]
    water_model = general_config["water model"]
    cutoff = general_config["LJ interaction cutoff"]*nanometer

    ######## DEFAULT - OPENFF ###############
    if ff_type == "openff":

        from openff.toolkit.typing.engines.smirnoff import ForceField as OFFForceField

        # Define the openff forcefield (by default 2.0.0)
        off_ff = OFFForceField("openff-2.0.0.offxml")

        # Create the system from the forcefield and topology object
        system = off_ff.create_openmm_system(topology)

        return system


    ######## AMBER #############
    elif ff_type == "amber":

        # Pull the amber forcefield xml first
        amber_files = ["amber14-all.xml"]

        # Depending on water configuration, add tip3 or tip4 xmls, or none
        if water_model == "tip3":
            amber_files.append("tip3p.xml")
        elif water_model == "tip4":
            amber_files.append("tip4pew.xml")

        # Add the various forcefield files to a forcefield object
        ff = ForceField(*amber_files)

        # Convert this to an openmm topology object
        openmm_top = topology.to_openmm()

        # Try default amber - this will work for proteins
        try:
            print("Trying default AMBER first")
            system = ff.createSystem(openmm_top, nonbondedMethod=PME, nonbondedCutoff=cutoff, constraints=HBonds)
            print("System compatible with default AMBER, succeeded")
            return system

        except Exception as e:
            print("System incompatible with default AMBER. Trying GAFF AMBER")
            print("Reason:", e)

        # Use the GAFF version of amber - will work for small organic molecules
        if molecules is None:
            raise ValueError("GAFF requires 'molecules' (a list of openff Molecule objects)")

        # Create and register the GAFF template generator
        gaff = GAFFTemplateGenerator(molecules=molecules)
        ff.registerTemplateGenerator(gaff.generator)

        # Create the GAFF system
        system = ff.createSystem(openmm_top, nonbondedMethod=PME, nonbondedCutoff=cutoff, constraints=HBonds)

        print("GAFF version of AMBER succeeded")
        return system


    else:
        raise ValueError("Force field type not supported")
    

def run_aamd_simulation(system, topology, positions, temperature, pressure, simulation_type, general_config, identifier, project_name, trajectory_path, save_diagnostics=False):
    """
    Run an all atom MD simulation for the defined system configuration

    Parameters
    ----------
    system : openmm system object
        Defines the system as a whole, including forcefield
    topology : openmm topology object
        Defines the topology of the system
    positions : list, openmm positions
        Defines positions of the molecules after packing
    temperature : float
        Temperature of the system, in Kelvin
    pressure : float
        Pressure of the system, in bars
    simulation_type : str
        Type of the simulation, either NPT or NVT
    general_config : dict
        Defines other configurable parameters of the system needed for the simulation
    identifier : str
        Unique identifying name for the pdb, inp, and trajectory file
    project_name : str
        Name of the project corresponding to this system / simulation
    trajectory_path : str
        Path to output the trajectory file to
    save_diagnostics : bool
        Whether to save logged parameters of simulation temperature, density, volume, etc. 
,
    Returns
    -------
    None
        Outputs simulation trajectory file
    """

    # Define a langevin integrator (acts as thermostat), with temperature, friction, and timestep
    friction = general_config["friction"]
    timestep = general_config["integration timestep"]
    integrator = LangevinIntegrator(temperature, friction/picosecond, timestep*femtoseconds)

    # If the system is NPT, set a barostat
    if simulation_type == "NPT":
        barostat_frequency_steps = steps_from_ns(general_config["pressure enforcing frequency"], general_config["integration timestep"])
        barostat = MonteCarloBarostat(pressure * bar, temperature, barostat_frequency_steps)
        system.addForce(barostat)

    # Create a simulation object using topology, system and integrator
    simulation = Simulation(topology.to_openmm(), system, integrator)
    simulation.context.setPositions(positions)

    # First, minimize the system (regardless of NPT or NVT)
    print("Minimizing system")
    simulation.minimizeEnergy()

    # Get necessary config params to calculate step intervals
    timestep_fs = general_config["integration timestep"]
    equil_time_ns = general_config["equilibration time"]
    prod_time_ns = general_config["production time"]
    traj_freq_ns = general_config["trajectory log frequency"]

    # Calculate step quantities for equilibration, production, and reporting
    equil_steps = compute_md_steps(equil_time_ns, timestep_fs)
    prod_steps = compute_md_steps(prod_time_ns, timestep_fs)
    report_interval = compute_report_interval(traj_freq_ns, timestep_fs)

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
        nvt_equil_steps = steps_from_ps(100.0, general_config["integration timestep"])
        simulation.step(nvt_equil_steps) # total 100 ps, constant (not in config)

        # Set the barostat back on and perform full NPT equilibration
        print("NPT equilibration and density relaxation")
        simulation.context.setParameter(MonteCarloBarostat.Pressure(), pressure * bar) # set barostat back on
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(equil_steps)

    # Add reporters for production
    simulation.reporters.append(PDBReporter(trajectory_path, report_interval))
    simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, temperature=True, density=True)) # another constant, always prints out at frequency of 5000 steps
    if save_diagnostics:
        log_path = project_name + '/' + identifier + "_log.txt"
        print(f"Reported quantities from simulation (T, E, \u03C1, V) will be written to the file: {log_path}")
        simulation.reporters.append(StateDataReporter(log_path, report_interval, step=True, temperature=True, 
                                                      potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                                      density=True, volume=True))

    # Run production
    print("Running production")
    simulation.step(prod_steps)
