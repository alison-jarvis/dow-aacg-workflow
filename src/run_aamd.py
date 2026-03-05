from AllAtomSimulation import AllAtomSimulation
import argparse
from pathlib import Path

def search_for_config(config_name, config_type):
    # Expected folders for types of files
    if config_type == 'general':
        expected_folder = './general_config_files/'
    elif config_type == 'system':
        expected_folder = './system_config_files/'

    # Look for config in its expected folder
    file_pattern = config_name + '.config'
    config_files = list(Path(expected_folder).glob(file_pattern))
    # If not found, search in the main folder
    if not config_files:
        general_folder = './'
        config_files = list(Path(general_folder).glob(file_pattern))
        if not config_files:
            raise Exception(f"No file names within the repository were found to match the pattern {file_pattern}")
        
    return str(config_files[0])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to run an AAMD simulation given config files.")
    parser.add_argument("system", type=str, help="The system config file name.")
    parser.add_argument("general", type=str, help="The general config file name.")
    args = parser.parse_args()

    system_config_name = args.system
    general_config_name = args.general

    system_config_path = search_for_config(system_config_name, 'system')
    general_config_path = search_for_config(general_config_name, 'general')

    # Define all atom simulation object
    aasim = AllAtomSimulation(system_config_path, general_config_path)

    # Run a simulation
    aasim.run_simulation()