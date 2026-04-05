# Imports
import re
import os
from aamd_utils import *
from cgmd_utils import *

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
        self.general_config = parse_general_config(general_config_path)

        # Generate topology and create system
        self.generate_topology()
        self.create_system()

    def generate_topology(self):
        # Find the relevant CG pdb path (asssumes in project folder as cg_start.pdb)
        cg_trajectory_path = self.project_name + '/cg_start.pdb'

        # Search for this path, and raise error if it doesn't exist
        if not os.path.exists(cg_trajectory_path):
            raise Exception(f"Coarse grained starting file doesn't exist for project {self.project_name}. Please run an all atom simulation first.")
        
        # Create topology object from this input
        self.topology, self.positions = generate_cg_topology(cg_trajectory_path)

    def create_system(self):
        # Create the system
        self.system = create_cg_system(self.topology, self.positions, self.project_name, 
                                       self.identifier, self.general_config, self.temperature)

    def run_simulation(self, iteration_number = 0, save_diagnostics = True):
        # Make (and clear) folder for cg results within project
        cg_trajectory_path = "./" + self.project_name + "/cg_trajectories/"
        if (os.path.exists(cg_trajectory_path)):
            os.system("rm -rf " + cg_trajectory_path)
            os.system("mkdir " + cg_trajectory_path)
        else:
            os.system("mkdir " + cg_trajectory_path)

        # If you're outputting the logs, also create a log folder
        if save_diagnostics:
            cg_diagnostic_path = "./" + self.project_name + "/cg_diagnostics/"
            if (os.path.exists(cg_diagnostic_path)):
                os.system("rm -rf " + cg_diagnostic_path)
                os.system("mkdir " + cg_diagnostic_path)
            else:
                os.system("mkdir " + cg_diagnostic_path)

        # Run a simulation
        run_cg_simulation(self.topology, self.positions, self.system, self.general_config, 
                          self.project_name, self.temperature, self.simulation_type, self.pressure, 
                          iteration_number=iteration_number, save_diagnostics=save_diagnostics)