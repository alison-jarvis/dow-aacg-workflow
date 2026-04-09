# Imports
from validation_utils import *

class SimulationValidation():

    def __init__(self, project_name):
        # Initialize attributes
        self.project_name = project_name

        # Create a validation directory within project
        validation_directory = self.project_name + "/validation/"
        os.makedirs(validation_directory, exist_ok=True)


    # Plot the CG minimization evolution
    def visualize_cg_minimization(self):
        visualize_cg_minimization(self.project_name)

    # Plot the all atom diagnostics
    def visualize_aamd_diagnostics(self):
        plot_aamd_diagnostics(self.project_name)

    # Plot the coarse grain diagnostics
    def visualize_cmgd_diagnostics(self):
        plot_cgmd_diagnostics(self.project_name)

    # Visualize final rdfs (aa vs cg)
    def visualize_rdfs(self):
        plot_final_rdfs(self.project_name)