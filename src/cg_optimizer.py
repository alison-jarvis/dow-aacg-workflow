import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import openmm as omm
import openmm.app as app
import openmm.unit as unit

class SRelOptimizer:
    def __init__(self, project_dir, pdb_path, rdf_csv_path, mscg_binary="/opt/conda/envs/mscg_engine/bin/cgrem"):
        self.project_dir = Path(project_dir)
        self.pdb_path = self.project_dir / pdb_path
        self.rdf_csv_path = self.project_dir / rdf_csv_path
        self.mscg_binary = mscg_binary

        self.temperature = 300 #kelvin
        self.kT = 0.00831446 * self.temperature #kJ/mol

        self.pairs = []
        self.r_grid = None

    def _generate_initial_guesses(self):
        """Reads target RDFs and uses Direct Boltzmann Inversion to create the best guess B-spline tables for OpenMM"""
        print("Generating initial potentions with DBI")

        df = pd.read_csv(self.rdf_csv_path)
        self.r_grid = df['r'].values / 10.0 #convert from angstrom to nm

        self.pairs = [col for col in df.columns if '-' in col]

        for pair in self.pairs:
            target_rdf = df[pair].values

            #DBI math: U(r) = -kT * ln(g(r))
            safe_rdf = np.where(target_rdf < 1e-5, 1e-5, target_rdf) #prevents explosions
            potential = -self.kT * np.log(safe_rdf)
            potential -= potential[-1]

            #expected MSCG table is [r, U, force]
            forces = np.gradient(-potential, self.r_grid)

            table_data = np.column_stack((self.r_grid, potential, forces))
            table_path = self.project_dir / f"{pair}.table"
            np.savetxt(table_path, table_data, header="r U F")
            print(f"Wrote initial guess: {table_path.name}")

    def _generate_rem_inp(self):
        """Generates the rem.inp config file for OpenMSCG to define B-spline resolution and cutoffs for each pair"""
        inp_path = self.project_dir / "rem.inp"
        r_min = self.r_grid[0]
        r_max = self.r_grid[-1]

        with open(inp_path, "w") as f:
            for pair in self.pairs:
                t1, t2 = pair.split('-')
                f.write(f"model Pair --type {t1},{t2} --min {r_min:.2f} --max {r_max:.2f} --resolution 0.05 --order 3\n")
        print(f"Generated rem.inp file: {inp_path.name}")

    def _run_openmm_step(self, iteration, steps=50000):
        """Loads B-spline tables and runs MD simulation"""
        print(f"Running OpenMM CG Simulation (Iter {iteration})")
        pdb = app.PDBFile(str(self.pdb_path))
        system = omm.System()

        # Add particles to the system
        for _ in pdb.topology.atoms():
            system.addParticle(10 * unit.amu)
        
        system.setDefaultPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

        # ==========================================
        # MULTI-PAIR TABULATED FORCE SETUP
        # ==========================================
        
        # 1. Identify unique bead types from self.pairs
        unique_types = []
        for pair in self.pairs:
            t1, t2 = pair.split('-')
            if t1 not in unique_types: unique_types.append(t1)
            if t2 not in unique_types: unique_types.append(t2)
            
        N_types = len(unique_types)
        type_to_idx = {name: i for i, name in enumerate(unique_types)}
        
        # 2. Build mapping structures
        pair_matrix = [0] * (N_types * N_types)
        table_data_list = []
        r_grid = None
        
        # 3. Load the correct table for every pair
        for pair_idx_counter, pair in enumerate(self.pairs):
            t1, t2 = pair.split('-')
            idx1 = type_to_idx[t1]
            idx2 = type_to_idx[t2]
            
            # Map both symmetric directions to the same pair ID
            pair_matrix[idx1 + idx2 * N_types] = pair_idx_counter
            pair_matrix[idx2 + idx1 * N_types] = pair_idx_counter
            
            # Determine which file to load based on the iteration
            if iteration == 0:
                table_path = self.project_dir / f"{pair}.table"
            else:
                # OpenMSCG usually names outputs like 'Pair_A_B.table'
                table_path = self.project_dir / f"Pair_{t1}_{t2}.table"
                if not table_path.exists():
                    table_path = self.project_dir / f"Pair_{t2}_{t1}.table"
                    
            if not table_path.exists():
                raise FileNotFoundError(f"Could not find table for {pair} at {table_path}")
                
            data = np.loadtxt(table_path)
            if r_grid is None:
                r_grid = data[:, 0]
            
            potential = data[:, 1]
            table_data_list.extend(potential.tolist()) # Flatten into 1D array for OpenMM

        # 4. Define the 2D Tabulated Force
        N_pairs = len(self.pairs)
        xsize = len(r_grid)
        ysize = N_pairs
        
        energy_expr = "tabulated_energy(r, pair_idx(type1, type2))"
        force = omm.CustomNonbondedForce(energy_expr)
        
        # Map (type1, type2) -> pair_idx
        force.addTabulatedFunction("pair_idx", omm.Discrete2DFunction(N_types, N_types, pair_matrix))
        
        # Map (r, pair_idx) -> Energy
        # Note: y bounds for Continuous2DFunction must be [-0.5, size - 0.5]
        force.addTabulatedFunction("tabulated_energy", omm.Continuous2DFunction(
            xsize, ysize, table_data_list, r_grid[0], r_grid[-1], -0.5, ysize - 0.5
        ))
        
        force.addPerParticleParameter("type")

        # 5. Safe Cutoff Logic (Minimum Image Convention)
        force.setNonbondedMethod(omm.CustomNonbondedForce.CutoffPeriodic)
        box_vectors = pdb.topology.getPeriodicBoxVectors()
        min_box_length = min([
            box_vectors[0][0].value_in_unit(unit.nanometers),
            box_vectors[1][1].value_in_unit(unit.nanometers),
            box_vectors[2][2].value_in_unit(unit.nanometers)
        ])
        
        max_allowed_cutoff = (min_box_length / 2.0) * 0.99
        actual_cutoff = min(r_grid[-1], max_allowed_cutoff)
        force.setCutoffDistance(actual_cutoff * unit.nanometers)

        # 6. Assign the integer 'type' parameter to each atom
        for atom in pdb.topology.atoms():
            atom_type = f"{atom.residue.name.strip()}_{atom.name.strip()}"
            if atom_type not in type_to_idx:
                raise ValueError(f"Atom type {atom_type} not found in RDF pairs!")
            force.addParticle([type_to_idx[atom_type]])
            
        system.addForce(force)

        # ==========================================
        # RUN SIMULATION
        # ==========================================
        integrator = omm.LangevinMiddleIntegrator(self.temperature * unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        simulation.minimizeEnergy()
        traj_path = str(self.project_dir / f"iter_{iteration}.dcd")
        
        # Note: Fixed 'repporters' typo here!
        simulation.reporters.append(app.DCDReporter(traj_path, 1000))
        simulation.step(steps)
        print(f"    -> Saved trajectory: iter_{iteration}.dcd")

    def _run_mscg_step(self, iteration):
        print(f"Calling openmscg to minimize (Iteration {iteration})")

        # 1. Generate an OpenMSCG-compatible topology (.top) from the PDB
        import openmm.app as app
        pdb = app.PDBFile(str(self.pdb_path))
        atoms = list(pdb.topology.atoms())
        cgsites = len(atoms)
        
        # Extract the same types we use in _run_openmm_step
        types_list = [f"{atom.residue.name.strip()}_{atom.name.strip()}" for atom in atoms]
        unique_types = list(dict.fromkeys(types_list)) # Removes duplicates, keeps order
            
        top_file = self.project_dir / "cg.top"
        with open(top_file, "w") as f:
            f.write(f"cgsites {cgsites}\n")
            f.write(f"cgtypes {len(unique_types)}\n")
            for t in unique_types:
                f.write(f"{t}\n")
            
            # Treat the entire system as one giant "molecule" to satisfy MSCG
            f.write("moltypes 1\n")
            f.write(f"mol {cgsites} 0\n")
            f.write("sitetypes\n")
            for t in types_list:
                f.write(f"{t}\n")
            f.write("bonds 0\n")
            f.write("system 1\n")
            f.write("1 1\n") # 1 instance of molecule type 1

        # 2. Write the cgderiv arguments, NOW POINTING TO cg.top
        cgderiv_arg_file = self.project_dir / "cgderiv.inp"
        with open(cgderiv_arg_file, "w") as f:
            f.write(f"--top cg.top --traj iter_{iteration}.dcd\n")

        # 3. Call OpenMSCG
        cmd = [
            self.mscg_binary,
            "--ref", "target_distributions/",            
            "--models", "rem.inp",                       
            "--cgderiv-arg", "cgderiv.inp",              
            "--md", "echo 'MD handled by Python'",       
            "--maxiter", "1"                            
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_dir)

        if result.returncode != 0:
            print(f"OpenMSCG Error:\n{result.stderr}")
            raise RuntimeError(f"Srel minimization failed on iteration {iteration}")

        print(f"    -> B-spline tables for iteration {iteration} optimized by OpenMSCG")

    def optimize(self, iterations = 20, steps_per_iter = 50000):
        print("Optimizing...")
        self._generate_initial_guesses()
        self._generate_rem_inp()

        for i in range(iterations):
            print(f"\n Iteration: {i+1}/{iterations}")
            self._run_openmm_step(iteration = i, steps = steps_per_iter)
            self._run_mscg_step(iteration = i)
        
        print("Optimization complete")

if __name__ == "__main__":
    project_directory = "./Project_Mixed_NVT"
    optimizer = SRelOptimizer(project_directory,
    pdb_path = "cg_start.pdb",
    rdf_csv_path = "cg_rdfs.csv",
    mscg_binary = "/opt/conda/envs/mscg_engine/bin/cgrem"
    )

    optimizer.optimize(iterations = 20, steps_per_iter = 50000)