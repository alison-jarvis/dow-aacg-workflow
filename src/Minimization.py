from CoarseGrainSimulation import CoarseGrainSimulation
from aamd_utils import *
from minimization_utils import *
from BoxPacker import BoxPacker
import csv

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
        self.aa_positions, self.aa_boxes, self.aa_bead_defs = map_aa_trajectory_to_cg_arrays(
            topology_path=aa_topology_path,
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
        iterations = int(self.general_config["opt iterations"])
        threshold = float(self.general_config["opt threshold"])
        learning_rate = float(self.general_config["opt learning rate"])

        print(f"=== Starting CG Parameter Optimization ({iterations} Iterations) ===")

        parameter_directory = self.project_name + "/cg_parameters/"
        os.makedirs(parameter_directory, exist_ok=True)

        rdf_errors = [["Iteration", "RDF Error"]]
        srel_history = [["Iteration", "Gradient Norm"]]
        h2o_grad_history = [["Iteration", "H2O_H2O_Gradient", "H2O_H2O_Gamma"]]
        a_history = [["Iteration", "H2O_a", "H2O_a_grad"]]
        srel_approx_history = [["Iteration", "Approx_Srel", "Mean_U_AA", "Var_U_AA"]]

        for i in range(iterations):
            print(f"\n--- Iteration {i} ---")

            # 1. Run CGMD with current parameters
            self.cg_sim.run_simulation(iteration_number=i, save_diagnostics=True)

            # 2. Load current CG trajectory
            self.load_current_cg_iteration(i)

            # 3. Compute Srel gradient
            gradients = calculate_srel_gradients(
                aa_positions=self.aa_positions,
                cg_positions=self.cg_positions,
                topology=self.cg_sim.topology,
                parameter_set=self.cg_sim.parameters,
                aa_boxes=self.aa_boxes,
                cg_boxes=self.cg_boxes,
                cutoff_nm=self.general_config["cg interaction cutoff"],
                frame_stride=1
            )

            # 4. Approximate Srel diagnostic
            srel_tilde, U_mean, U_var, U_frames = calculate_approx_srel_nvt(
                positions=self.aa_positions,
                topology=self.cg_sim.topology,
                parameter_set=self.cg_sim.parameters,
                boxes=self.aa_boxes,
                cutoff_nm=self.general_config["cg interaction cutoff"],
                frame_stride=1
            )

            srel_approx_history.append([i, srel_tilde, U_mean, U_var])
            print(f"Approx Srel (AA-mapped frames): {srel_tilde:.6e}")
            print(f"Mean U_CG on AA frames: {U_mean:.6e}")
            print(f"Var U_CG on AA frames: {U_var:.6e}")

            # 5. Log H2O gamma and a before update
            h2o_key = ("H2O_B1", "H2O_B1")
            h2o_type = "H2O_B1"

            h2o_grad = gradients["pair"]["gamma"].get(h2o_key, None)
            h2o_gamma = self.cg_sim.parameters.pair_parameters["gamma"].get(h2o_key, None)

            h2o_a = self.cg_sim.parameters.get_individual(h2o_type, "a")
            h2o_a_grad = gradients["individual"]["a"].get(h2o_type, None)

            h2o_grad_history.append([i, h2o_grad, h2o_gamma])
            a_history.append([i, h2o_a, h2o_a_grad])

            print(f"H2O gamma: {h2o_gamma:.6e}")
            print(f"H2O gamma grad: {h2o_grad:.6e}")
            print(f"H2O a: {h2o_a:.6e}")
            print(f"H2O a grad: {h2o_a_grad:.6e}")

            # Optional: freeze b if desired
            # for key in gradients["pair"].get("b", {}):
            #     gradients["pair"]["b"][key] = 0.0

            # 6. Compute convergence metric
            grad_norm = calculate_srel_gradient_norm(gradients)
            srel_history.append([i, grad_norm])
            print(f"Srel gradient norm: {grad_norm:.6e}")

            # 7. RDF diagnostic
            self.current_rdfs = calculate_current_rdfs(i, self.project_name)
            current_error = calculate_rdf_error(self.target_rdfs, self.current_rdfs)
            rdf_errors.append([i, current_error])
            print(f"RDF diagnostic error: {current_error:.6e}")

            # 8. Save current parameters
            write_intermediate_parameters(self.cg_sim.parameters, self.project_name, i)

            # 9. Check convergence
            if grad_norm < threshold:
                print(f"Srel minimization converged in {i} iterations.")
                break

            # 10. Print top gamma gradients
            gamma_grads = gradients["pair"]["gamma"]
            sorted_items = sorted(gamma_grads.items(), key=lambda kv: abs(kv[1]), reverse=True)

            print("Top 10 |gamma gradients|:")
            for key, val in sorted_items[:10]:
                old_val = self.cg_sim.parameters.pair_parameters["gamma"][key]
                expected_delta = -learning_rate * val
                print(
                    f"{key}: grad={val:.6e}, "
                    f"gamma={old_val:.6e}, "
                    f"expected_delta={expected_delta:.6e}"
                )

            # 11. Gamma and a update checks before applying gradients
            old_gamma = self.cg_sim.parameters.pair_parameters["gamma"][h2o_key]
            gamma_grad = gradients["pair"]["gamma"][h2o_key]
            expected_new_gamma = old_gamma - learning_rate * gamma_grad

            old_a = self.cg_sim.parameters.get_individual(h2o_type, "a")
            a_grad = gradients["individual"]["a"][h2o_type]
            expected_new_a = old_a - learning_rate * a_grad

            print("=== H2O gamma sign check ===")
            print(f"old gamma:          {old_gamma:.12e}")
            print(f"gradient:           {gamma_grad:.12e}")
            print(f"learning rate:      {learning_rate:.12e}")
            print(f"expected new gamma: {expected_new_gamma:.12e}")

            print("=== H2O width (a) check ===")
            print(f"old a:              {old_a:.12e}")
            print(f"a grad:             {a_grad:.12e}")
            print(f"expected new a:     {expected_new_a:.12e}")

            # 12. Apply gradients
            self.cg_sim.parameters.apply_gradients(gradients, learning_rate)

            # 13. Log actual updated values
            actual_new_gamma = self.cg_sim.parameters.pair_parameters["gamma"][h2o_key]
            actual_new_a = self.cg_sim.parameters.get_individual(h2o_type, "a")

            print(f"actual new gamma:   {actual_new_gamma:.12e}")
            print(f"gamma difference:   {actual_new_gamma - expected_new_gamma:.12e}")

            print(f"actual new a:       {actual_new_a:.12e}")
            print(f"a difference:       {actual_new_a - expected_new_a:.12e}")
            print("============================")

            print("gamma sample after update:",
                list(self.cg_sim.parameters.pair_parameters["gamma"].items())[:3])
            print(f"updated a: {actual_new_a:.6e}")

            # 14. Rebuild OpenMM system with updated parameters
            self.cg_sim.update_system()

        # Finalization
        with open(self.project_name + "/rdf_errors.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rdf_errors)

        with open(self.project_name + "/srel_gradient_norms.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(srel_history)

        with open(self.project_name + "/h2o_h2o_gradients.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(h2o_grad_history)

        with open(self.project_name + "/a_history.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(a_history)

        with open(self.project_name + "/srel_approximate.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(srel_approx_history)

        write_final_parameters(self.cg_sim.parameters, self.project_name)

        print("Optimization completed.")
        export_rdfs_to_csv(self.target_rdfs, self.current_rdfs, self.project_name)