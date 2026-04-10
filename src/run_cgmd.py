from CoarseGrainSimulation import CoarseGrainSimulation
import argparse
from run_aamd import search_for_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to run a CGMD simulation given config files.")
    parser.add_argument("system", type=str, help="The system config file name.")
    parser.add_argument("general", type=str, help="The general config file name.")
    args = parser.parse_args()

    system_config_name = args.system
    general_config_name = args.general

    system_config_path = search_for_config(system_config_name, 'system')
    general_config_path = search_for_config(general_config_name, 'general')

    # Define coarse grained simulation object
    cgsim = CoarseGrainSimulation(system_config_path, general_config_path)

    # Run a simulation
    cgsim.optimize(iterations=100)