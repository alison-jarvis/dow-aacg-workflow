from ValidateSimulation import SimulationValidation
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to run validation for a project.")
    parser.add_argument("project_name", type=str, help="The project name to validate.")
    args = parser.parse_args()

    project_name = args.project_name

    # Define a validation object
    valobj = SimulationValidation(project_name)

    # Run all of the methods
    valobj.visualize_cg_minimization()
    valobj.visualize_aamd_diagnostics()
    valobj.visualize_cmgd_diagnostics()
    valobj.visualize_rdfs()
