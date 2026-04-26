import pandas as pd
import numpy as np
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from openmm.unit import kilojoule_per_mole
import itertools
from cg_build import *
from cgmd_utils import *
import os
import csv


############### RDF Utils #################

# Calculate rdfs for the current step
def calculate_current_rdfs(iteration_number, project_name):
    #import current trajectory, topology
    trajectory_path = f"{project_name}/cg_trajectories/cgmd_trajectory_{iteration_number}.dcd"
    topology_path = f"{project_name}/cg_start.pdb"

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

# Calculate error between AA and CG RDFs
def calculate_rdf_error(target_rdfs, current_rdfs):
    
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

# Export the rdfs to a csv file
def export_rdfs_to_csv(target_rdfs, current_rdfs, project_name, output_filename="final_rdfs.csv"):
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
    output_path = f"{project_name}/{output_filename}"
    df.to_csv(output_path, index=False)
    print(f"\nSaved final RDFs to {output_path}")


################## Srel Utils ####################

# Helper - mirror across box
def minimum_image(delta, box_lengths):
    return delta - box_lengths * np.round(delta / box_lengths)

# Helper - get bead type pairs for nonbonded
def get_nonbonded_index_pairs(topology):
    n_atoms = topology.getNumAtoms()
    bonded = set()
    for bond in topology.bonds():
        i = bond[0].index
        j = bond[1].index
        bonded.add(tuple(sorted((i, j))))

    pairs = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if (i, j) not in bonded:
                pairs.append((i, j))
    return np.asarray(pairs, dtype=int)

# Helper - get bead type pairs for bonded
def get_bond_index_pairs(topology):
    return np.asarray([(bond[0].index, bond[1].index) for bond in topology.bonds()], dtype=int)

# Helper - bead type by index
def get_bead_types_by_index(topology):
    bead_types = {}
    for atom in topology.atoms():
        bead_types[atom.index] = bead_type_id(atom)
    return bead_types

# Helper - initialize a gradient dictionary
def init_gradient_dict(parameter_set):
    gradients = {"pair": {pname: {} for pname in parameter_set.pair_names},
                 "individual": {pname: {} for pname in parameter_set.individual_names},
                 "fixed": {pname: 0.0 for pname in parameter_set.fixed_names}}

    for pname, pmap in parameter_set.pair_parameters.items():
        for key in pmap.keys():
            gradients["pair"][pname][key] = 0.0

    for pname, pmap in parameter_set.individual_parameters.items():
        for key in pmap.keys():
            gradients["individual"][pname][key] = 0.0

    return gradients

# Nonbonded srel gradient accumulation
def accumulate_nonbonded_srel(
    positions, boxes, topology, parameter_set,
    bead_types_by_index, frame_stride, cutoff_nm=None):
    """
    Accumulate ensemble averages of nonbonded derivatives for Srel.

    U = kBT * gamma * exp(-r^2 / (2*(a1^2 + a2^2)))

    Returns <dU/dlambda> averaged over frames.
    """
    kBT = parameter_set.get_fixed("kBT")
    try:
        kBT_val = kBT.value_in_unit(kilojoule_per_mole)
    except Exception:
        kBT_val = float(kBT)

    acc = init_gradient_dict(parameter_set)
    frame_count = 0

    pair_indices = get_nonbonded_index_pairs(topology)
    if len(pair_indices) == 0:
        return acc

    ii = pair_indices[:, 0]
    jj = pair_indices[:, 1]
    use_pbc = boxes is not None

    pair_keys = []
    bead_type_i = []
    bead_type_j = []
    gamma_arr = []
    a1_arr = []
    a2_arr = []

    for i, j in pair_indices:
        bt_i = bead_types_by_index[i]
        bt_j = bead_types_by_index[j]
        pair_key = canonical_pair(bt_i, bt_j)

        pair_keys.append(pair_key)
        bead_type_i.append(bt_i)
        bead_type_j.append(bt_j)

        gamma_arr.append(parameter_set.get_pair(pair_key, "gamma"))
        a1_arr.append(parameter_set.get_individual(bt_i, "a"))
        a2_arr.append(parameter_set.get_individual(bt_j, "a"))

    gamma_arr = np.asarray(gamma_arr, dtype=float)
    a1_arr = np.asarray(a1_arr, dtype=float)
    a2_arr = np.asarray(a2_arr, dtype=float)

    for frame_idx in range(0, positions.shape[0], frame_stride):
        pos = positions[frame_idx]

        delta = pos[ii] - pos[jj]
        if use_pbc:
            delta = minimum_image(delta, boxes[frame_idx])

        r = np.linalg.norm(delta, axis=1)

        if cutoff_nm is not None:
            mask = r < cutoff_nm
            if not np.any(mask):
                continue

            indices = np.where(mask)[0]
            r_use = r[mask]
            gamma_use = gamma_arr[mask]
            a1_use = a1_arr[mask]
            a2_use = a2_arr[mask]
        else:
            indices = np.arange(len(r))
            r_use = r
            gamma_use = gamma_arr
            a1_use = a1_arr
            a2_use = a2_arr

        D = a1_use * a1_use + a2_use * a2_use
        exp_term = np.exp(-(r_use * r_use) / (2.0 * D))

        U = kBT_val * gamma_use * exp_term

        d_gamma = kBT_val * exp_term

        rr = r_use * r_use
        D2 = D * D

        d_a1 = U * rr * a1_use / D2
        d_a2 = U * rr * a2_use / D2

        frame_pair_gamma = {}
        frame_indiv_a = {}

        for local_k, global_k in enumerate(indices):
            pair_key = pair_keys[global_k]
            bt_i = bead_type_i[global_k]
            bt_j = bead_type_j[global_k]

            frame_pair_gamma[pair_key] = frame_pair_gamma.get(pair_key, 0.0) + d_gamma[local_k]
            frame_indiv_a[bt_i] = frame_indiv_a.get(bt_i, 0.0) + d_a1[local_k]
            frame_indiv_a[bt_j] = frame_indiv_a.get(bt_j, 0.0) + d_a2[local_k]

        for key, val in frame_pair_gamma.items():
            acc["pair"]["gamma"][key] += val

        for key, val in frame_indiv_a.items():
            acc["individual"]["a"][key] += val

        frame_count += 1

    n = max(frame_count, 1)

    for key in acc["pair"]["gamma"]:
        acc["pair"]["gamma"][key] /= n

    for key in acc["individual"]["a"]:
        acc["individual"]["a"][key] /= n

    return acc

# Bonded srel gradient accumulation
def accumulate_bonded_srel(positions, boxes, topology, parameter_set, bead_types_by_index, frame_stride):
    """
    Accumulate ensemble averages of bonded derivatives for srel force
        U = (3*kBT/(2*b^2)) * r^2
        dU/db = -(3*kBT/b^3) * r^2
    """
    # Initialize gradient dict and bond pair indices
    acc = init_gradient_dict(parameter_set)
    bond_indices = get_bond_index_pairs(topology)
    if len(bond_indices) == 0:
        return acc

    # Indices for bonded pairs specifically
    ii = bond_indices[:, 0]
    jj = bond_indices[:, 1]

    use_pbc = boxes is not None

    pair_keys = []
    b_arr = []

    # Get the value of the boltzmann constant
    kBT = parameter_set.get_fixed("kBT")
    try:
        kBT_val = kBT.value_in_unit(kilojoule_per_mole)
    except Exception:
        kBT_val = float(kBT)

    # Compile values of b for each pair (bond)
    for i, j in bond_indices:
        bt_i = bead_types_by_index[i]
        bt_j = bead_types_by_index[j]
        pair_key = canonical_pair(bt_i, bt_j)

        pair_keys.append(pair_key)
        b_arr.append(parameter_set.get_pair(pair_key, "b"))

    b_arr = np.asarray(b_arr, dtype=float)

    frame_count = 0

    # Iterate through frames of trajectory
    for frame_idx in range(0, positions.shape[0], frame_stride):
        pos = positions[frame_idx]
        delta = pos[ii] - pos[jj]

        # Get radii for pairs, accounting for pbcs
        if use_pbc:
            box = boxes[frame_idx]
            delta = minimum_image(delta, box)

        r = np.linalg.norm(delta, axis=1)

        # dU/db = -(3*kBT/b^3) * r^2
        d_b = -(3.0 * kBT_val / (b_arr ** 3)) * (r * r)

        # Get the change in the b parameter for pairs
        for k, pair_key in enumerate(pair_keys):
            acc["pair"]["b"][pair_key] += d_b[k]

        frame_count += 1

    n = max(frame_count, 1)

    # Take the average
    for key in acc["pair"]["b"]:
        acc["pair"]["b"][key] /= n

    return acc

# Calculate gradient dictionary for srel
def calculate_srel_gradients(aa_positions, cg_positions, topology, parameter_set, aa_boxes=None, 
                             cg_boxes=None, scale_by_beta=True, cutoff_nm=None, frame_stride=1):
    """
    Fast REM-style gradient calculator specialized to the current SREL forcefield.

    Parameters
    ----------
    aa/cg_positions : np.ndarray
        Shape (n_frames, n_beads, 3), mapped AA->CG or CG positions in nm
    topology : openmm topology object
        CG topology
    parameter_set : ParameterSet
        Parameters for the gradient
    aa/cg_boxes : np.ndarray or None
        Box lengths for each frame in nm
    scale_by_beta : bool
        If true, multiply final gradients by beta = 1/kBT
    cutoff_nm : float or None
        Nonbonded cutoff in nm. If None, no cutoff applied in gradient accumulation
    frame_stride : int
        For potential speedup, can use every n frames

    Returns
    -------
    gradients : dict
        Dictionary mapping delta theta to theta (parameter update -> parameter)
    """

    # Get bead types
    bead_types_by_index = get_bead_types_by_index(topology)

    # Accumulate nonbonded for AA and CG
    aa_nb = accumulate_nonbonded_srel(aa_positions, aa_boxes, topology, parameter_set, bead_types_by_index, frame_stride, cutoff_nm)
    cg_nb = accumulate_nonbonded_srel(cg_positions, cg_boxes, topology, parameter_set, bead_types_by_index, frame_stride, cutoff_nm)

    # Accumulate bonded for AA and CG
    aa_bd = accumulate_bonded_srel(aa_positions, aa_boxes, topology, parameter_set, bead_types_by_index, frame_stride)
    cg_bd = accumulate_bonded_srel(cg_positions, cg_boxes, topology, parameter_set, bead_types_by_index, frame_stride)

    # Initialize a gradient dictionary from parameters
    gradients = init_gradient_dict(parameter_set)

    # Calculate gradient for pair parameters as (AA_NB + AA_B) - (CG_NB + CG_B)
    for pname in gradients["pair"]:
        for key in gradients["pair"][pname]:
            gradients["pair"][pname][key] = (
                aa_nb["pair"].get(pname, {}).get(key, 0.0)
                + aa_bd["pair"].get(pname, {}).get(key, 0.0)
                - cg_nb["pair"].get(pname, {}).get(key, 0.0)
                - cg_bd["pair"].get(pname, {}).get(key, 0.0))

    # Calculate gradient for individual parameters as (AA_NB + AA_B) - (CG_NB + CG_B)
    for pname in gradients["individual"]:
        for key in gradients["individual"][pname]:
            gradients["individual"][pname][key] = (
                aa_nb["individual"].get(pname, {}).get(key, 0.0)
                + aa_bd["individual"].get(pname, {}).get(key, 0.0)
                - cg_nb["individual"].get(pname, {}).get(key, 0.0)
                - cg_bd["individual"].get(pname, {}).get(key, 0.0))

    # Calculate gradient for fixed parameters as (AA_NB + AA_B) - (CG_NB + CG_B)
    for pname in gradients["fixed"]:
        gradients["fixed"][pname] = (
            aa_nb["fixed"].get(pname, 0.0)
            + aa_bd["fixed"].get(pname, 0.0)
            - cg_nb["fixed"].get(pname, 0.0)
            - cg_bd["fixed"].get(pname, 0.0))

    # Multiply the gradient by the boltzmann constant (optional)
    if scale_by_beta:
        kBT = parameter_set.get_fixed("kBT")
        try:
            beta = 1.0 / kBT.value_in_unit(kilojoule_per_mole)
        except Exception:
            beta = 1.0 / float(kBT)

        for pname in gradients["pair"]:
            for key in gradients["pair"][pname]:
                gradients["pair"][pname][key] *= beta

        for pname in gradients["individual"]:
            for key in gradients["individual"][pname]:
                gradients["individual"][pname][key] *= beta

        for pname in gradients["fixed"]:
            gradients["fixed"][pname] *= beta

    return gradients


############## Trajectory Loading Utils ##############

def load_cg_trajectory_arrays(cg_topology_path, cg_trajectory_path, start=None, stop=None, 
                              step=None):
    """
    Load an already coarse grained trajectory into arrays for gradient evaluation

    Returns
    -------
    cg_positions_nm : np.ndarray
        Shape (n_frames, n_beads, 3), in nm
    cg_boxes_nm : np.ndarray
        Shape (n_frames, 3), in nm
    """

    # MDA universe from topology and trajectory
    universe = mda.Universe(cg_topology_path, cg_trajectory_path)

    traj = universe.trajectory[start:stop:step]
    n_frames = len(traj)
    n_beads = len(universe.atoms)

    positions_ang = np.zeros((n_frames, n_beads, 3), dtype=np.float64)
    boxes_ang = np.zeros((n_frames, 3), dtype=np.float64)

    for fi, ts in enumerate(traj):
        positions_ang[fi] = universe.atoms.positions.copy()
        boxes_ang[fi] = ts.dimensions[:3].copy()

    # Convert angstrom to nm
    cg_positions_nm = positions_ang * 0.1
    cg_boxes_nm = boxes_ang * 0.1

    return cg_positions_nm, cg_boxes_nm

def map_aa_trajectory_to_cg_arrays(topology_path, trajectory_path, mapping_rules, start=None, 
                                   stop=None, step=None):
    """
    Map an all atom trajectory into CG bead coordinates

    Returns
    -------
    aa_positions_nm : np.ndarray
        Shape (n_frames, n_beads, 3), in nm
    aa_boxes_nm : np.ndarray
        Shape (n_frames, 3), in nm
    bead_defs : list
        The bead definitions produced by build_universal_bead_mapping
    """

    universe = mda.Universe(topology_path, trajectory_path)

    # Keep molecules whole across periodic boundaries before COM calculation
    workflow = [trans.unwrap(universe.atoms)]
    universe.trajectory.add_transformations(*workflow)

    # Build CG mapping once
    bead_defs = build_universal_bead_mapping(universe, mapping_rules)

    # Compute COM positions for all requested frames
    positions_ang, box_lengths_ang = compute_bead_positions(universe, bead_defs, start=start, 
                                                            stop=stop, step=step)

    # Convert angstrom to nm for the optimizer
    aa_positions_nm = positions_ang * 0.1
    aa_boxes_nm = box_lengths_ang * 0.1

    return aa_positions_nm, aa_boxes_nm, bead_defs

def flatten_gradient_dict(gradients):
    vals = []

    for pname in sorted(gradients["pair"].keys()):
        for key in sorted(gradients["pair"][pname].keys()):
            vals.append(float(gradients["pair"][pname][key]))

    for pname in sorted(gradients["individual"].keys()):
        for key in sorted(gradients["individual"][pname].keys()):
            vals.append(float(gradients["individual"][pname][key]))

    for pname in sorted(gradients["fixed"].keys()):
        vals.append(float(gradients["fixed"][pname]))

    return np.array(vals, dtype=float)

def calculate_srel_gradient_norm(gradients):
    g = flatten_gradient_dict(gradients)
    if g.size == 0:
        return 0.0
    return float(np.linalg.norm(g))

def compute_cg_frame_energies(
    positions,
    topology,
    parameter_set,
    boxes=None,
    cutoff_nm=None,
    frame_stride=1):
    """
    Compute total CG potential energy U_CG for each frame.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n_frames, n_beads, 3), coordinates in nm.
    topology : OpenMM topology
        CG topology.
    parameter_set : ParameterSet
        Current CG parameters.
    boxes : np.ndarray or None
        Shape (n_frames, 3), box lengths in nm.
    cutoff_nm : float or None
        Nonbonded cutoff in nm.
    frame_stride : int
        Use every nth frame.

    Returns
    -------
    energies : np.ndarray
        1D array of total CG energies per frame, in kJ/mol if kBT is in kJ/mol.
    """
    bead_types_by_index = get_bead_types_by_index(topology)

    kBT_value = parameter_set.get_fixed("kBT")
    try:
        kBT_val = kBT_value.value_in_unit(kilojoule_per_mole)
    except Exception:
        kBT_val = float(kBT_value)


    # ---------- nonbonded precompute ----------
    nb_pairs = get_nonbonded_index_pairs(topology)
    use_pbc = boxes is not None

    if len(nb_pairs) > 0:
        nb_ii = nb_pairs[:, 0]
        nb_jj = nb_pairs[:, 1]

        nb_gamma = []
        nb_a1 = []
        nb_a2 = []

        for i, j in nb_pairs:
            bt_i = bead_types_by_index[i]
            bt_j = bead_types_by_index[j]
            pair_key = canonical_pair(bt_i, bt_j)

            nb_gamma.append(parameter_set.get_pair(pair_key, "gamma"))
            nb_a1.append(parameter_set.get_individual(bt_i, "a"))
            nb_a2.append(parameter_set.get_individual(bt_j, "a"))

        nb_gamma = np.asarray(nb_gamma, dtype=float)
        nb_a1 = np.asarray(nb_a1, dtype=float)
        nb_a2 = np.asarray(nb_a2, dtype=float)
    else:
        nb_ii = nb_jj = None

    # ---------- bonded precompute ----------
    bd_pairs = get_bond_index_pairs(topology)

    if len(bd_pairs) > 0:
        bd_ii = bd_pairs[:, 0]
        bd_jj = bd_pairs[:, 1]

        bd_b = []
        for i, j in bd_pairs:
            bt_i = bead_types_by_index[i]
            bt_j = bead_types_by_index[j]
            pair_key = canonical_pair(bt_i, bt_j)
            bd_b.append(parameter_set.get_pair(pair_key, "b"))

        bd_b = np.asarray(bd_b, dtype=float)

        kBT = parameter_set.get_fixed("kBT")
        try:
            kBT_val = kBT.value_in_unit(kilojoule_per_mole)
        except Exception:
            kBT_val = float(kBT)
    else:
        bd_ii = bd_jj = None
        bd_b = None

    energies = []

    for frame_idx in range(0, positions.shape[0], frame_stride):
        pos = positions[frame_idx]
        U_frame = 0.0

        # ---------- nonbonded energy ----------
        if len(nb_pairs) > 0:
            delta = pos[nb_ii] - pos[nb_jj]
            if use_pbc:
                box = boxes[frame_idx]
                delta = minimum_image(delta, box)

            r = np.linalg.norm(delta, axis=1)

            if cutoff_nm is not None:
                mask = r < cutoff_nm
                if np.any(mask):
                    r_use = r[mask]
                    gamma_use = nb_gamma[mask]
                    a1_use = nb_a1[mask]
                    a2_use = nb_a2[mask]

                    D = a1_use * a1_use + a2_use * a2_use
                    exp_term = np.exp(-(r_use * r_use) / (2.0 * D))
                    U_nb = np.sum(kBT_val * gamma_use * exp_term)
                    U_frame += U_nb
            else:
                D = nb_a1 * nb_a1 + nb_a2 * nb_a2
                exp_term = np.exp(-(r * r) / (2.0 * D))
                U_nb = np.sum(kBT_val * nb_gamma * exp_term)
                U_frame += U_nb

        # ---------- bonded energy ----------
        if len(bd_pairs) > 0:
            delta = pos[bd_ii] - pos[bd_jj]
            if use_pbc:
                box = boxes[frame_idx]
                delta = minimum_image(delta, box)

            r = np.linalg.norm(delta, axis=1)
            U_bd = np.sum(kBT_val * (3.0 / (2.0 * bd_b * bd_b)) * (r * r))
            U_frame += U_bd

        energies.append(U_frame)

    return np.asarray(energies, dtype=float)

def calculate_approx_srel_nvt(
    positions,
    topology,
    parameter_set,
    boxes=None,
    cutoff_nm=None,
    frame_stride=1
    ):
    """
    Approximate relative entropy in NVT:
        Srel_tilde = 0.5 * beta^2 * var(U_CG)

    Parameters
    ----------
    positions : np.ndarray
        Shape (n_frames, n_beads, 3), typically AA-mapped CG coordinates.
    topology : OpenMM topology
        CG topology.
    parameter_set : ParameterSet
        Current CG parameters.
    boxes : np.ndarray or None
        Shape (n_frames, 3), box lengths in nm.
    cutoff_nm : float or None
        Nonbonded cutoff in nm.
    frame_stride : int
        Use every nth frame.

    Returns
    -------
    srel_tilde : float
        Approximate relative entropy estimate.
    energy_mean : float
        Mean CG energy over sampled frames.
    energy_var : float
        Variance of CG energy over sampled frames.
    energies : np.ndarray
        Per-frame CG energies.
    """
    energies = compute_cg_frame_energies(
        positions=positions,
        topology=topology,
        parameter_set=parameter_set,
        boxes=boxes,
        cutoff_nm=cutoff_nm,
        frame_stride=frame_stride
    )

    if len(energies) == 0:
        return 0.0, 0.0, 0.0, energies

    energy_mean = float(np.mean(energies))
    energy_var = float(np.var(energies, ddof=0))

    kBT = parameter_set.get_fixed("kBT")
    try:
        beta = 1.0 / kBT.value_in_unit(kilojoule_per_mole)
    except Exception:
        beta = 1.0 / float(kBT)

    srel_tilde = 0.5 * (beta ** 2) * energy_var

    return float(srel_tilde), energy_mean, energy_var, energies


################ Csv Writing Utils ###################


def write_parameter_set_to_csv(parameter_set, output_dir):
    """
    Write parameter object to 3 csv files:
        - pair_parameters.csv
        - individual_parameters.csv
        - fixed_parameters.csv

    Parameters
    ----------
    parameter_set : ParameterSet
    output_dir : str
        Directory to write csv files into
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. Pairwise parameters
    pair_file = os.path.join(output_dir, "pair_parameters.csv")

    if parameter_set.pair_parameters:
        # Collect all unique pair keys
        all_pairs = set()
        for pname in parameter_set.pair_names:
            all_pairs.update(parameter_set.pair_parameters[pname].keys())

        all_pairs = sorted(all_pairs)

        with open(pair_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["bead_type_i", "bead_type_j"] + parameter_set.pair_names
            writer.writerow(header)

            # Rows
            for pair in all_pairs:
                row = [pair[0], pair[1]]

                for pname in parameter_set.pair_names:
                    val = parameter_set.pair_parameters.get(pname, {}).get(pair, "")
                    row.append(val)

                writer.writerow(row)

    # 2. Individual parameters
    indiv_file = os.path.join(output_dir, "individual_parameters.csv")

    if parameter_set.individual_parameters:
        all_bead_types = set()
        for pname in parameter_set.individual_names:
            all_bead_types.update(parameter_set.individual_parameters[pname].keys())

        all_bead_types = sorted(all_bead_types)

        with open(indiv_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["bead_type"] + parameter_set.individual_names
            writer.writerow(header)

            # Rows
            for bt in all_bead_types:
                row = [bt]

                for pname in parameter_set.individual_names:
                    val = parameter_set.individual_parameters.get(pname, {}).get(bt, "")
                    row.append(val)

                writer.writerow(row)

    # 3. Fixed parameters
    fixed_file = os.path.join(output_dir, "fixed_parameters.csv")

    if parameter_set.fixed_parameters:
        with open(fixed_file, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["parameter", "value"])

            for pname in parameter_set.fixed_names:
                val = parameter_set.fixed_parameters.get(pname, "")

                # Handle OpenMM quantities nicely
                try:
                    val = float(val)
                except Exception:
                    try:
                        val = val.value_in_unit(kilojoule_per_mole)
                    except Exception:
                        pass

                writer.writerow([pname, val])

# Helper - write intermediate parameters to a csv
def write_intermediate_parameters(parameter_set, project_name, iteration):
    base_dir = os.path.join(project_name, "cg_parameters", f"iteration_{iteration}")
    write_parameter_set_to_csv(parameter_set, base_dir)

# Helper - write final parameters to a csv
def write_final_parameters(parameter_set, project_name):
    final_dir = os.path.join(project_name, "final_parameters")
    write_parameter_set_to_csv(parameter_set, final_dir)

