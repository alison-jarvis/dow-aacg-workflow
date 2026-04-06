# Imports
import re
import os
from aamd_utils import *
import MDAnalysis as mda
from cg_build import rdf_for_pair
from openmm.app import PDBFile, Simulation, PDBReporter, DCDReporter, StateDataReporter
from openmm import CustomNonbondedForce, HarmonicBondForce, System, LangevinIntegrator, MonteCarloBarostat
from openmm.unit import dalton, nanometer, kelvin, picosecond, femtoseconds, bar, norm
from openmm.unit import BOLTZMANN_CONSTANT_kB, MOLAR_GAS_CONSTANT_R
from pathlib import Path
from sys import stdout
import itertools
import pandas as pd
import numpy as np

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

        self.target_rdfs = pd.read_csv(f"{self.project_name}/cg_rdfs.csv")
        

        # Generate topology and create system
        self.generate_topology()

        self.initialize_bead_params()

        self.create_system()

    def generate_topology(self):
        # Find the relevant CG pdb path (asssumes in project folder as cg_start.pdb)
        cg_trajectory_path = self.project_name + '/cg_start.pdb'

        # Search for this path, and raise error if it doesn't exist
        if not os.path.exists(cg_trajectory_path):
            raise Exception(f"Coarse grained starting file doesn't exist for project {self.project_name}. Please run an all atom simulation first.")
        
        # Create topology object from this input
        pdb = PDBFile(cg_trajectory_path)
        self.topology = pdb.topology
        self.positions = pdb.positions

    def initialize_bead_params(self):
        """Extracts unique composite IDs and sets initial gamma and a."""
        self.bead_params = {}
        initial_gamma = self.general_config["default parameter 1"]
        initial_a = self.general_config["default parameter 2"]

        for atom in self.topology.atoms():
            # Create composite ID safely: e.g., "C8H-B1"
            unique_id = f"{atom.residue.name.strip()}_{atom.name.strip()}"
            if unique_id not in self.bead_params:
                self.bead_params[unique_id] = [initial_gamma, initial_a]

    def create_system(self):
        self.system = System()

        # Set periodic box vectors
        inp_path = f"{self.project_name}/{self.identifier}.inp"
        pbc_margin = self.general_config["periodic box margin"]
        box_vectors = parse_packmol_box(inp_path, pbc_margin=pbc_margin)
        self.system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

        # Add particles
        for atom in self.topology.atoms():
            self.system.addParticle(54*dalton) 

        # Srel Forcefield
        if self.general_config["cg forcefield"] == "srel":
            # Using mixing rules: sqrt(gamma1 * gamma2)
            gaussian_force = CustomNonbondedForce("""sqrt(gamma1*gamma2) * exp(-r^2 / (2*(a1*a1 + a2*a2)))""")
            gaussian_force.addPerParticleParameter("gamma")
            gaussian_force.addPerParticleParameter("a")
            gaussian_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
            gaussian_force.setCutoffDistance(self.general_config["cg interaction cutoff"]*nanometer)

            # Assign parameters using tracked composite IDs
            for atom in self.topology.atoms():
                unique_id = f"{atom.residue.name.strip()}_{atom.name.strip()}"
                gamma_bead, a_bead = self.bead_params[unique_id] 
                gaussian_force.addParticle([gamma_bead, a_bead]) 

            # Exclude bonded pairs
            bonds = [(bond[0].index, bond[1].index) for bond in self.topology.bonds()]
            gaussian_force.createExclusionsFromBonds(bonds, 1)
            self.system.addForce(gaussian_force)

            # Harmonic Bonds
            bond_force = HarmonicBondForce()
            b = self.general_config["default parameter 3"]*nanometer 
            kBT = MOLAR_GAS_CONSTANT_R * self.temperature * kelvin
            k = 3*kBT/(b**2)

            for bond in self.topology.bonds():
                i, j = bond[0].index, bond[1].index
                delta = self.positions[i] - self.positions[j]
                bond_force.addBond(i, j, norm(delta), k)

            self.system.addForce(bond_force)

    def run_simulation(self, iteration_number = 0, save_diagnostics = True):
        # Folder management
        traj_dir = f"./{self.project_name}/cg_trajectories/"
        diag_dir = f"./{self.project_name}/cg_diagnostics/"
        os.makedirs(traj_dir, exist_ok=True)
        if save_diagnostics:
            os.makedirs(diag_dir, exist_ok=True)

        integrator = LangevinIntegrator(
            self.temperature, 
            self.general_config["cg friction"]/picosecond, 
            self.general_config["cg integration timestep"]*femtoseconds
        )

        if self.simulation_type == "NPT":
            barostat_freq = steps_from_ns(self.general_config["cg pressure enforcing frequency"], self.general_config["cg integration timestep"])
            barostat = MonteCarloBarostat(self.pressure * bar, self.temperature, barostat_freq)
            self.system.addForce(barostat)

        simulation = Simulation(self.topology, self.system, integrator)
        simulation.context.setPositions(self.positions)

        print(f"Minimizing system for iteration {iteration_number}...")
        simulation.minimizeEnergy()

        timestep_fs = self.general_config["cg integration timestep"]
        equil_steps = compute_md_steps(self.general_config["cg equilibration time"], timestep_fs)
        prod_steps = compute_md_steps(self.general_config["cg production time"], timestep_fs)
        report_interval = compute_report_interval(self.general_config["cg trajectory log frequency"], timestep_fs)

        # Equilibration
        if self.simulation_type == "NVT":
            simulation.context.setVelocitiesToTemperature(self.temperature)
            simulation.step(equil_steps)
        elif self.simulation_type == "NPT":
            simulation.context.setVelocitiesToTemperature(self.temperature)
            simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0 * bar) 
            simulation.step(steps_from_ps(100.0, timestep_fs)) 
            simulation.context.setParameter(MonteCarloBarostat.Pressure(), self.pressure * bar) 
            simulation.context.setVelocitiesToTemperature(self.temperature)
            simulation.step(equil_steps)

        # Production Reporters
        traj_path = f"{traj_dir}/cgmd_trajectory_{iteration_number}.dcd"
        simulation.reporters.append(DCDReporter(traj_path, report_interval))
        simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, temperature=True, density=True))

        if save_diagnostics:
            log_path = f"{diag_dir}/cg_log_{iteration_number}.txt"
            simulation.reporters.append(StateDataReporter(log_path, report_interval, step=True, temperature=True, potentialEnergy=True, density=True, volume=True))

        print(f"Running production for iteration {iteration_number}...")
        simulation.step(prod_steps)
        
        # Save the final positions to update the starting point for the next iteration
        self.positions = simulation.context.getState(getPositions=True).getPositions()

    def calculate_current_rdfs(self, iteration_number):
        #import current trajectory, topology
        trajectory_path = f"{self.project_name}/cg_trajectories/cgmd_trajectory_{iteration_number}.dcd"
        topology_path = f"{self.project_name}/cg_start.pdb"

        #generate universe
        print(f"  -> Calculating RDFs for Iteration {iteration_number}...")
        u = mda.Universe(topology_path, trajectory_path)

        #generate rdf labels
        bead_ids = [f"{atom.resname.strip()}_{atom.name.strip()}" for atom in u.atoms]

        type_to_indices = {}
        for i, b_id in enumerate(bead_ids):
            type_to_indices.setdefault(b_id, []).append(i)
            
        unique_bead_ids = list(type_to_indices.keys())
        pair_list = list(itertools.combinations_with_replacement(unique_bead_ids, 2))

        #get positions and boxes
        all_positions = []
        all_boxes = []
        
        for ts in u.trajectory:
            all_positions.append(u.atoms.positions.copy())
            all_boxes.append(ts.dimensions[:3].copy())

        positions_array = np.array(all_positions)
        boxes_array = np.array(all_boxes)

        rdf_results = {}
        r_grid = None
        
        #make rdf list
        for a, b in pair_list:
            r, g = rdf_for_pair(
                positions_array,
                boxes_array,
                type_to_indices[a],
                type_to_indices[b],
                r_max=20.0,
                n_bins=200
            )
            if r_grid is None:
                r_grid = r
                rdf_results["r"] = r_grid
                
            rdf_results[f"{a}-{b}"] = g

        return pd.DataFrame(rdf_results)
    
    def update_gaussian_parameters(self, target_rdfs, current_rdfs, lr_gamma=0.1, lr_alpha=0.01):
        """
        Relative Entropy gradient-descent update rule for analytical Gaussian parameters.
        """
        import numpy as np
        print("  -> Updating Gamma and Alpha parameters using S_rel gradients...")
        
        # Assuming 'r' is a column in your DataFrame. If it's the index, use target_rdfs.index.values
        r = target_rdfs['r'].values 
        
        for bead_type in self.bead_params.keys():
            current_gamma = self.bead_params[bead_type][0]
            current_a = self.bead_params[bead_type][1]
            
            pair_key = f"{bead_type}-{bead_type}" 
            
            if pair_key in target_rdfs.columns and pair_key in current_rdfs.columns:
                g_target = target_rdfs[pair_key].values
                g_current = current_rdfs[pair_key].values
                
                # The RDF error: if CG is over-structured, error is positive.
                error = g_current - g_target
                
                # 1. Calculate Gradients
                # Gradient wrt gamma: exp(-alpha * r^2)
                grad_gamma = np.exp(-current_a * r**2)
                
                # Gradient wrt alpha: -gamma * r^2 * exp(-alpha * r^2)
                grad_alpha = -current_gamma * (r**2) * np.exp(-current_a * r**2)
                
                # 2. Integrate (Sum) the error weighted by the gradients
                # We multiply the error by the gradient at each 'r' bin, and sum them up.
                delta_gamma = np.sum(error * grad_gamma)
                delta_alpha = np.sum(error * grad_alpha)
                
                # 3. Apply Updates
                new_gamma = current_gamma + (lr_gamma * delta_gamma)
                new_a = current_a + (lr_alpha * delta_alpha)
                
                # 4. Enforce Physical Boundaries
                # Gamma should usually be positive (repulsive core)
                new_gamma = max(0.01, new_gamma)
                # Alpha MUST be positive, otherwise the Gaussian explodes to infinity at long distances!
                new_a = max(0.1, new_a)
                
                self.bead_params[bead_type] = [new_gamma, new_a]
                print(f"     {bead_type}:")
                print(f"       \u03B3 updated: {current_gamma:.4f} -> {new_gamma:.4f}")
                print(f"       \u03B1 updated: {current_a:.4f} -> {new_a:.4f}")
            else:
                print(f"Pair key not found: {pair_key}")


    def calculate_rdf_error(self, target_rdfs, current_rdfs):
        import numpy as np
        
        total_error = 0.0
        pair_count = 0

        # Identify the distance column so we don't calculate error on the X-axis
        r_col = 'r' if 'r' in target_rdfs.columns else target_rdfs.columns[0]

        for pair_name in target_rdfs.columns:
            if pair_name == r_col:
                continue # Skip the distance bins!

            if pair_name in current_rdfs.columns:
                # Extract the FULL arrays using .values
                target_gr = target_rdfs[pair_name].values
                current_gr = current_rdfs[pair_name].values

                # Calculates the mean error across all distance bins
                pair_error = np.mean(np.abs(target_gr - current_gr))
                total_error += pair_error
                pair_count += 1
        
        if pair_count == 0:
            return float('inf') # Prevent division by zero if nothing matches
            
        return total_error / pair_count

    def export_rdfs_to_csv(self, target_rdfs, current_rdfs, output_filename="final_rdfs.csv"):
        # 1. Identify your distance column (usually 'r' or 'r_nm' in the DataFrame)
        r_col = 'r' if 'r' in target_rdfs.columns else target_rdfs.columns[0]
        
        # Extract the entire array of distances
        data_dict = {"r_nm": target_rdfs[r_col].values}
        
        # 2. Loop through the columns to match target and current pairs
        for pair in target_rdfs.columns:
            if pair == r_col: 
                continue # Skip the distance column
                
            if pair in current_rdfs.columns:
                # Use .values to extract the full numpy array of the column
                data_dict[f"{pair}_target"] = target_rdfs[pair].values
                data_dict[f"{pair}_current"] = current_rdfs[pair].values
                
        # 3. Create the DataFrame (No brackets around data_dict!)
        df = pd.DataFrame(data_dict)
        
        # 4. Save to CSV
        output_path = f"{self.project_name}/{output_filename}"
        df.to_csv(output_path, index=False)
        print(f"\nSaved final RDFs to {output_path}")

    def optimize(self, iterations=50, threshold = 0.05):
        """
        The master loop that iterates the Gaussian system toward the target RDFs.
        """
        print(f"=== Starting Gaussian Parameter Optimization ({iterations} Iterations) ===")
        
        # 2. Initialize our parameter tracking dictionary if it doesn't exist
        if not hasattr(self, 'bead_params'):
            initial_gamma = self.general_config["default parameter 1"]
            initial_a = self.general_config["default parameter 2"]
            self.bead_params = build_bead_params_srel(self.topology, initial_gamma, initial_a)

        # 3. The Loop
        for i in range(iterations):
            print(f"\n--- Iteration {i} ---")
            
            # Re-create the OpenMM system with the NEW parameters
            self.create_system() 
            
            # Run the simulation 
            self.run_simulation(iteration_number=i, save_diagnostics=True)
            
            # Calculate the resulting structure
            self.current_rdfs = self.calculate_current_rdfs(iteration_number=i)
            current_error = self.calculate_rdf_error(self.target_rdfs, self.current_rdfs)

            if current_error <= threshold:
                print(f"Convergence threshold reached after {i} iterations")
                break
            # Adjust the parameters for the next loop
            self.update_gaussian_parameters(self.target_rdfs, self.current_rdfs)
            
        print("Optimization completed.")
        self.export_rdfs_to_csv(self.target_rdfs, self.current_rdfs)