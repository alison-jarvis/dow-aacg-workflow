from CoarseGrainSimulation import CoarseGrainSimulation
from aamd_utils import *
from minimization_utils import *
from BoxPacker import BoxPacker

class CG_AA_Minimizer():

    def __init__(self, system_config_path, general_config_path):
        # Add the paths to the configs as attributes
        self.system_config_path = system_config_path
        self.general_config_path = general_config_path

        # Determine project identifier
        self.identifier = re.search(r'[^/]+(?=\.config)', system_config_path)[0]

        # Get the general config file as a dictionary
        self.general_config, self.ff_params = parse_general_config(general_config_path, with_forcefield=True)

        # Set the pressure, temperature, simulation type, and project name from system config
        self.project_name, self.pressure, self.temperature, self.simulation_type = parse_simulation_parameters(system_config_path)

        # Initialize bead mapping rules as attribute
        box_pack = BoxPacker(self.system_config_path, overwrite=True, filename = self.identifier)
        self.mapping_rules = box_pack.mapping_rules

        # Initialize the coarse-grained all atom reference data
        aa_trajectory_path = self.project_name + '/' + self.identifier + '.dcd'
        aa_topology_path = self.project_name + '/' + self.identifier + '.pdb'
        self.initialize_reference_cg_data(aa_topology_path, aa_trajectory_path)

        # Read the target (AA) rdfs from the starting csv
        self.target_rdfs = pd.read_csv(f"{self.project_name}/cg_rdfs.csv")

        # Initialize with starting CG object
        self.cg_sim = CoarseGrainSimulation(self.system_config_path, self.general_config_path)

    def initialize_reference_cg_data(self, aa_topology_path, aa_trajectory_path, aa_start=None, aa_stop=None, aa_step=None):
        """
        Build and store the mapped AA to CG trajectory arrays only once
        """
        # Construct all atom positions / box vectors / bead definitions from aa trajectory
        self.aa_positions, self.aa_boxes, self.aa_bead_defs = \
            map_aa_trajectory_to_cg_arrays(topology_path=aa_topology_path, 
                                           trajectory_path=aa_trajectory_path, 
                                           mapping_rules=self.mapping_rules, 
                                           start=aa_start,
                                           stop=aa_stop,
                                           step=aa_step)

    def load_current_cg_iteration(self, iteration_number, cg_start=None, cg_stop=None, cg_step=None):
        # Define topology and trajectory for this iteration
        cg_topology_path = f"{self.project_name}/cg_start.pdb"
        cg_trajectory_path = f"{self.project_name}/cg_trajectories/cgmd_trajectory_{iteration_number}.dcd"

        # Convert these into positions / box vectors
        self.cg_positions, self.cg_boxes = load_cg_trajectory_arrays(cg_topology_path, 
                                                                     cg_trajectory_path, 
                                                                     start=cg_start, 
                                                                     stop=cg_stop, 
                                                                     step=cg_step)

    def minimize(self):
        # Read iterations, threshold and LR from config
        max_iterations = int(self.general_config["opt max iterations"])
        threshold = self.general_config["opt threshold"]
        learning_rate = self.general_config["opt learning rate"]

        print(f"=== Starting CG Parameter Optimization ({max_iterations} Iterations) ===")

        # Create directory for parameters
        parameter_directory = self.project_name + "/cg_parameters/"
        os.makedirs(parameter_directory, exist_ok=True)

        # Create directory for gradients
        gradient_directory = self.project_name + "/cg_gradients/"
        os.makedirs(gradient_directory, exist_ok=True)

        # Create list to save rdf error values
        rdf_errors = [["Iteration", "RDF Error"]]

        ########### Full Minimization Loop ##############
        for i in range(max_iterations):
            print(f"\n--- Iteration {i} ---")

            # Run a simulation with the current cg object
            self.cg_sim.run_simulation(iteration_number=i, save_diagnostics=True)

            # Load the current cg iteration after sim complete
            self.load_current_cg_iteration(i)

            # Calculate rdfs from the simulation run
            self.current_rdfs = calculate_current_rdfs(i, self.project_name, self.mapping_rules)
            current_error = calculate_rdf_error(self.target_rdfs, self.current_rdfs)

            # Write a line to the csv for the rdf error
            rdf_errors.append([i, current_error])

            # If i = 0, save the starting rdfs for reference
            if i == 0:
                # Write out starting rdfs
                export_rdfs_to_csv(self.target_rdfs, self.current_rdfs, self.project_name, output_filename="starting_rdfs.csv")

            # Write current parameters to csv file
            write_intermediate_parameters(self.cg_sim.parameters, self.project_name, i)

            ########## Case 1 - Converged ############
            #if (current_error < threshold) or (i > iterations):
            if i > max_iterations:
                if i <= max_iterations:
                    print(f'Minimization converged in {i} iterations.')
                else:
                    print(f"Reached maximum number of iterations, did not converge.")
                break

            ########## Case 2 - Not Converged ########

            gradients = calculate_srel_gradients(aa_positions=self.aa_positions,
                                                 cg_positions=self.cg_positions,
                                                 topology=self.cg_sim.topology,
                                                 parameter_set=self.cg_sim.parameters,
                                                 force_spec=self.cg_sim.forcefield_form,
                                                 aa_boxes=self.aa_boxes,
                                                 cg_boxes=self.cg_boxes,
                                                 scale_by_beta=False,
                                                 cutoff_nm=self.general_config["cg interaction cutoff"],
                                                 frame_stride=1)
            
            # Write these gradients to a csv
            write_intermediate_gradients(gradients, self.project_name, i)

            # Apply gradients to the parameter object
            self.cg_sim.parameters.apply_gradients(gradients, learning_rate)

            # Update CG simulation object based on parameter difference
            self.cg_sim.update_system()

        ########## Post Loop Finalization ############

        # First, write out rdf errors
        rdf_csv_path = self.project_name + "/rdf_errors.csv"
        with open(rdf_csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rdf_errors)

        # Second, write out final parameters
        write_final_parameters(self.cg_sim.parameters, self.project_name)

        # Write out final rdfs
        print("Optimization completed.")
        export_rdfs_to_csv(self.target_rdfs, self.current_rdfs, self.project_name)