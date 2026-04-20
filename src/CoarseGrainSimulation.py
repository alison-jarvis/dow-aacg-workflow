# Imports
from aamd_utils import *
from cgmd_utils import *
from openmm.app import PDBFile

class CoarseGrainSimulation():

    def __init__(self, system_config_path, general_config_path):

        # Add the paths to the configs as attributes
        self.system_config_path = system_config_path
        self.general_config_path = general_config_path

        # Set the identifier from the simulation config name
        self.identifier = re.search(r'[^/]+(?=\.config)', system_config_path)[0]

        # Set the pressure, temperature, simulation type, and project name from system config
        self.project_name, self.pressure, self.temperature, self.simulation_type = parse_simulation_parameters(system_config_path)

        # Get the general config file as a dictionary
        self.general_config, ff_params = parse_general_config(general_config_path, with_forcefield=True)

        # Generate topology
        self.generate_topology()

        # Initialize the parameter object from the general config
        self.parameters = parse_parameters_from_config(ff_params, self.topology, self.temperature)
        # Initialize forcefield object from general config
        self.forcefield_form = parse_forcefield_from_config(self.general_config)

        # Set target radial distribution functions (beaded all atom rdfs)
        self.target_rdfs = pd.read_csv(f"{self.project_name}/cg_rdfs.csv")

        # Build bead mass mappings
        self.masses = extract_bead_mass_mapping(self.topology)

        # Create the initial system
        self.create_system()

    def generate_topology(self):
        # Find the relevant CG pdb path (asssumes in project folder as cg_start.pdb)
        cg_topology_path = self.project_name + '/cg_start.pdb'

        # Search for this path, and raise error if it doesn't exist
        if not os.path.exists(cg_topology_path):
            raise Exception(f"Coarse grained starting file doesn't exist for project {self.project_name}. Please run an all atom simulation first.")
        
        # Create topology object from this input
        pdb = PDBFile(cg_topology_path)
        self.topology = pdb.topology
        self.positions = pdb.positions


    def create_system(self):
        self.system = build_general_cg_system(self.topology, self.project_name, 
                                              self.identifier, self.masses, self.parameters, 
                                              self.forcefield_form, self.general_config)

    def run_simulation(self, iteration_number = 0, save_diagnostics = True):
        # Coarse grained simulation run
        new_positions = run_cg_simulation(self.topology, self.positions, self.system, 
                                          self.general_config, self.project_name, self.temperature, 
                                          self.simulation_type, self.pressure, 
                                          iteration_number=iteration_number, 
                                          save_diagnostics=save_diagnostics)
        
        # Save the final positions to update the starting point for the next iteration
        self.positions = new_positions

    def update_system(self):
        self.create_system()
