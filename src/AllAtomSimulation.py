from aamd_utils import *
from BoxPacker import BoxPacker

class AllAtomSimulation():

    def __init__(self, system_config_path, general_config_path):

        # Add the paths to the configs as attributes
        self.system_config_path = system_config_path
        self.general_config_path = general_config_path

        # Set the identifier from the simulation config name
        self.identifier = re.search(r'[^/]+(?=\.config)', system_config_path)[0]

        # Set the pressure, temperature, simulation type, and project name from system config
        self.project_name, self.pressure, self.temperature, self.simulation_type = parse_simulation_parameters(system_config_path)

        # Get the general config file as a dictionary
        self.general_config = parse_general_config(general_config_path)

        # Perform the setup to be able to run simulation - initialize, get topology and system attributes
        self.initialize_system()
        self.generate_topology()
        self.create_system()

    def initialize_system(self):
        # Initialize a box packer class, and pack with packmol
        box_pack = BoxPacker(self.system_config_path, overwrite=True, filename = self.identifier)
        box_pack.pack_the_mol()

    def generate_topology(self):
        # Generate the topology object, molecules, and positions
        self.topology, self.molecules, self.positions = generate_topology(self.system_config_path, self.identifier, self.project_name, pbc_margin = self.general_config["periodic box margin"])

    def create_system(self):
        # Create the system from topology, molecules, and general config parameters
        self.system = create_universal_system(self.topology, self.molecules, self.general_config)

    def run_simulation(self, trajectory_path=None, output_diagnostics=True):

        # User information for simulation, and reminder that pressure won't be used for NVT
        if self.simulation_type == "NVT":
            print(f"Running an NVT simulation for {self.project_name} at temperature {self.temperature} K. Pressure defined in the config file will not be used.")
        elif self.simulation_type == "NPT":
            print(f"Running an NPT simulation for {self.project_name} at pressure {self.pressure} bars and temperature {self.temperature} K.")
        else:
            raise Exception(f"Unknown simulation type {self.simulation_type}.")
        
        # Define trajectory name corresponding to identifier, if none provided
        if trajectory_path is None:
            trajectory_path = self.project_name + '/' + self.identifier + '.dcd'
        
        # Run the all atom simulation
        run_aamd_simulation(self.system, self.topology, self.positions, self.temperature, self.pressure, 
                            self.simulation_type, self.general_config, self.identifier, self.project_name, 
                            trajectory_path, save_diagnostics=output_diagnostics)
        
        print(f"Succesful AAMD run, produced trajectory file at {trajectory_path}")
        
