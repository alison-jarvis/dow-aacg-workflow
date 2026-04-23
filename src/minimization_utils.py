# Imports
import os
import csv
import pandas as pd
import numpy as np
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from openmm.unit import kilojoule_per_mole
import itertools
from collections import defaultdict
from cg_build import *
from cgmd_utils import *


############### RDF Utils #################

# Helper - RDF bead type labels
def current_rdf_bead_type(atom, mapping_rules=None):
    """
    Recover the CG bead type label used in target RDF generation, to replicate

    Parameters
    ----------
    atom : MDAnalysis atom
    mapping_rules : dict or None
        Same rules used in build_universal_bead_mapping()

    Returns
    -------
    bead_type : str
    """
    resname = atom.resname.strip()
    atomname = atom.name.strip()

    # Best / general path: use mapping rules
    if mapping_rules is not None:
        for _, rule in mapping_rules.items():
            cg_resname = rule.get("cg_resname", "")[:3]

            if cg_resname != resname:
                continue

            for bead_rule in rule["beads"]:
                if bead_rule["name"] == atomname:
                    return bead_rule["type"]

    # Fallbacks for older / hardcoded systems
    if resname == "WAT":
        return "WAT"
    if resname == "DIO":
        return "DIO"
    if resname == "DOD" and atomname.startswith("B"):
        # B1 -> D1, B2 -> D2, ...
        return f"D{atomname[1:]}"

    # Last fallback: just use residue name
    return resname

# Calculate rdfs for the current step
def calculate_current_rdfs(iteration_number, project_name, mapping_rules=None):
    # Load in current trajectory, topology
    trajectory_path = f"{project_name}/cg_trajectories/cgmd_trajectory_{iteration_number}.dcd"
    topology_path = f"{project_name}/cg_start.pdb"

    print(f"  -> Calculating RDFs for Iteration {iteration_number}...")
    u = mda.Universe(topology_path, trajectory_path)

    # Generate bead-type labels consistent with target RDF generation
    bead_ids = [current_rdf_bead_type(atom, mapping_rules) for atom in u.atoms]

    type_to_indices = {}
    for i, b_id in enumerate(bead_ids):
        type_to_indices.setdefault(b_id, []).append(i)

    # Keep pair ordering deterministic
    unique_bead_ids = sorted(type_to_indices.keys())
    pair_list = list(itertools.combinations_with_replacement(unique_bead_ids, 2))

    # Get positions and boxes
    all_positions = []
    all_boxes = []

    # Iterate through trajectory, get positions / boxes
    for ts in u.trajectory:
        all_positions.append(u.atoms.positions.copy())
        all_boxes.append(ts.dimensions[:3].copy())

    # Define positions and boxes
    positions_array = np.array(all_positions)
    boxes_array = np.array(all_boxes)

    # RDF results and grid
    rdf_results = {}
    r_grid = None

    # Get RDF for each pair
    for a, b in pair_list:
        r, g = rdf_for_pair(positions_array, boxes_array, type_to_indices[a], 
                            type_to_indices[b], r_max=20.0, n_bins=200)
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


############# General Minimization Utils ################

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

# Helper - get indices for pair keys in the overall keys
def pair_key_mask(pair_keys_all, pair_key):
    pair_keys_all = np.ravel(pair_keys_all)
    return np.array([pk == pair_key for pk in pair_keys_all], dtype=bool)

# Helper - build the list of array / constant arguments based on param order
def build_lambda_args(eval_param_names, values_dict):
    """
    Build ordered argument list for the lambdified derivative function
    """
    return [values_dict[name] for name in eval_param_names]


################ Srel Specific Utils #################

def compute_sampled_radii(positions, boxes, pair_indices, bead_types_by_index, frame_stride=1, cutoff_nm=None):
    """
    Compute sampled radii once for a set of index pairs across a trajectory

    Parameters
    ----------
    positions : np.ndarray
        Shape (n_frames, n_beads, 3), in nm
    boxes : np.ndarray or None
        Shape (n_frames, 3), in nm
    pair_indices : np.ndarray
        Shape (n_pairs, 2)
    bead_types_by_index : dict
        index -> bead_type string
    frame_stride : int
    cutoff_nm : float or None
        If given, keep only radii < cutoff_nm

    Returns
    -------
    samples : dict
        {
            "r": np.ndarray,
            "pair_keys": np.ndarray(dtype=object),   # 1d array of tuple objects
            "bead_type_i": np.ndarray(dtype=object), # 1d array of strings
            "bead_type_j": np.ndarray(dtype=object), # 1d array of strings
            "n_samples": int,
        }
    """
    # Return empty dictionary if none of these pairs
    if len(pair_indices) == 0:
        return {"r": np.empty(0, dtype=float), 
                "pair_keys": np.empty(0, dtype=object),
                "bead_type_i": np.empty(0, dtype=object),
                "bead_type_j": np.empty(0, dtype=object),
                "n_samples": 0}

    # Otherwise, initialize pair indices 
    pair_indices = np.asarray(pair_indices, dtype=int)
    ii = pair_indices[:, 0]
    jj = pair_indices[:, 1]

    # Whether to use pbc, defined by boxes
    use_pbc = boxes is not None

    # Build the 1d object arrays
    n_pairs = len(pair_indices)
    pair_keys = np.empty(n_pairs, dtype=object)
    bead_type_i = np.empty(n_pairs, dtype=object)
    bead_type_j = np.empty(n_pairs, dtype=object)

    # Fill these out from pair indices
    for k, (i, j) in enumerate(pair_indices):
        bt_i = bead_types_by_index[i]
        bt_j = bead_types_by_index[j]
        pair_keys[k] = canonical_pair(bt_i, bt_j)
        bead_type_i[k] = bt_i
        bead_type_j[k] = bt_j

    # Radii, keys, and bead types
    all_r, all_pair_keys, all_bt_i, all_bt_j = [], [], [], []

    # Iterate through frames in positions (at stride)
    for frame_idx in range(0, positions.shape[0], frame_stride):
        # Pull out the positions, get the difference
        pos = positions[frame_idx]
        delta = pos[ii] - pos[jj]

        # Calculate the new delta uses mirroring if pbc
        if use_pbc:
            box = boxes[frame_idx]
            delta = minimum_image(delta, box)

        # Radii as norm of difference
        r = np.linalg.norm(delta, axis=1)

        # Account for cutoff by masking radii
        if cutoff_nm is not None:
            mask = r < cutoff_nm
            if not np.any(mask):
                continue
            
            # Set these as cutoff arbitrary values
            frame_r = r[mask]
            frame_pair_keys = pair_keys[mask]
            frame_bt_i = bead_type_i[mask]
            frame_bt_j = bead_type_j[mask]
        else:
            frame_r = r
            frame_pair_keys = pair_keys
            frame_bt_i = bead_type_i
            frame_bt_j = bead_type_j

        # Append these values over all frames
        all_r.append(frame_r)
        all_pair_keys.append(frame_pair_keys)
        all_bt_i.append(frame_bt_i)
        all_bt_j.append(frame_bt_j)

    # Return empty again if no applicable radii
    if len(all_r) == 0:
        return {"r": np.empty(0, dtype=float),
                "pair_keys": np.empty(0, dtype=object),
                "bead_type_i": np.empty(0, dtype=object),
                "bead_type_j": np.empty(0, dtype=object),
                "n_samples": 0}

    # Otherwise, concatenate everything
    r_all = np.concatenate(all_r)
    pair_keys_all = np.concatenate(all_pair_keys)
    bt_i_all = np.concatenate(all_bt_i)
    bt_j_all = np.concatenate(all_bt_j)

    return {"r": r_all, "pair_keys": pair_keys_all, "bead_type_i": bt_i_all, 
            "bead_type_j": bt_j_all, "n_samples": len(r_all)}

# Function to accumulate nonbonded gradients
def accumulate_nonbonded_general(positions, boxes, topology, parameter_set, force_spec, bead_types_by_index,
                                 frame_stride=1, cutoff_nm=None):
    """
    Generalized nonbonded gradient accumulation using force_spec's nonbonded_first_derivatives

    Returns
    -------
    first_grad : dict
        Gradient dictionary, gradient for each parameter
    """
    # Initialize empty gradient / count dictionaries
    first_grad = init_gradient_dict(parameter_set)
    counts = init_gradient_dict(parameter_set)

    # Get the pair indices and get the radii
    pair_indices = get_nonbonded_index_pairs(topology)
    samples = compute_sampled_radii(positions=positions, boxes=boxes, pair_indices=pair_indices, 
                                    bead_types_by_index=bead_types_by_index, frame_stride=frame_stride,
                                    cutoff_nm=cutoff_nm)

    # Extract n samples, radii, and pair keys
    r_all = samples["r"]
    pair_keys_all = samples["pair_keys"]
    n_samples = samples["n_samples"]

    # If there are no samples, return gradient = 0 
    if n_samples == 0:
        return first_grad

    # Pairwise nonbonded parameters
    for pname in force_spec.nonbonded_pair_names:
        deriv_func = force_spec.nonbonded_first_derivatives[pname]

        # Iterate through pair keys
        for pair_key in parameter_set.pair_parameters[pname].keys():
            mask = pair_key_mask(pair_keys_all, pair_key)
            if not np.any(mask):
                continue
            
            # Mask the radii for that pair
            bt_i, bt_j = pair_key
            r = r_all[mask]

            # Initialize the values with array of r
            values = {"r": r}

            # All pairwise params are scalar constants for this pair type
            for qname in force_spec.nonbonded_pair_names:
                # Add constant to values for that parameter
                values[qname] = parameter_set.get_pair(pair_key, qname)

            # Individual params also scalar constants for this pair type
            for iname in force_spec.nonbonded_individual_names:
                values[f"{iname}1"] = parameter_set.get_individual(bt_i, iname)
                values[f"{iname}2"] = parameter_set.get_individual(bt_j, iname)

            # Fixed params are scalar constants as well
            for fname in force_spec.nonbonded_fixed_names:
                val = parameter_set.get_fixed(fname)
                try:
                    values[fname] = float(val)
                except Exception:
                    values[fname] = val.value_in_unit(kilojoule_per_mole)

            # Order the arguments correctly to pass into the lambda function
            args = build_lambda_args(force_spec.nonbonded_eval_params, values)
            contrib = deriv_func(*args)

            # Add the sum of the contribution over all radii to the gradient
            first_grad["pair"][pname][pair_key] += float(np.sum(contrib))
            # Also add counts, for later averaging
            counts["pair"][pname][pair_key] += len(r)

    # Individual nonbonded parameters
    for pname in force_spec.nonbonded_individual_names:
        deriv_func_1 = force_spec.nonbonded_first_derivatives[f"{pname}1"]
        deriv_func_2 = force_spec.nonbonded_first_derivatives[f"{pname}2"]

        # Get unique pair keys
        unique_pair_keys = set(pair_keys_all.tolist())

        # Iterate through each pair, mask it
        for pair_key in unique_pair_keys:
            mask = pair_key_mask(pair_keys_all, pair_key)
            if not np.any(mask):
                continue

            # Mask the radii for that pair key
            bt_i, bt_j = pair_key
            r = r_all[mask]

            # Add radii array to values
            values = {"r": r}

            # Add pair parameters as constants again
            for qname in force_spec.nonbonded_pair_names:
                values[qname] = parameter_set.get_pair(pair_key, qname)

            # Add individual parameters as constants
            for iname in force_spec.nonbonded_individual_names:
                values[f"{iname}1"] = parameter_set.get_individual(bt_i, iname)
                values[f"{iname}2"] = parameter_set.get_individual(bt_j, iname)

            # Add fixed parameters as constants
            for fname in force_spec.nonbonded_fixed_names:
                val = parameter_set.get_fixed(fname)
                try:
                    values[fname] = float(val)
                except Exception:
                    values[fname] = val.value_in_unit(kilojoule_per_mole)

            # Build the argument list for the lambda function
            args = build_lambda_args(force_spec.nonbonded_eval_params, values)

            # For pairs - get contribution for both type1, type2
            contrib_1 = deriv_func_1(*args)
            contrib_2 = deriv_func_2(*args)

            # Add sum of contribution to gradient and update counts
            first_grad["individual"][pname][bt_i] += float(np.sum(contrib_1))
            first_grad["individual"][pname][bt_j] += float(np.sum(contrib_2))
            counts["individual"][pname][bt_i] += len(r)
            counts["individual"][pname][bt_j] += len(r)

    # Fixed nonbonded parameters
    for pname in force_spec.nonbonded_fixed_names:
        deriv_func = force_spec.nonbonded_first_derivatives[pname]

        # Again, unique pair keys
        unique_pair_keys = set(pair_keys_all.tolist())

        # Mask out the radii for this unique pair
        for pair_key in unique_pair_keys:
            mask = pair_key_mask(pair_keys_all, pair_key)
            if not np.any(mask):
                continue

            bt_i, bt_j = pair_key
            r = r_all[mask]

            # Set array of radii in values
            values = {"r": r}

            # Pair values as constants
            for qname in force_spec.nonbonded_pair_names:
                values[qname] = parameter_set.get_pair(pair_key, qname)

            # Individual values as constants
            for iname in force_spec.nonbonded_individual_names:
                values[f"{iname}1"] = parameter_set.get_individual(bt_i, iname)
                values[f"{iname}2"] = parameter_set.get_individual(bt_j, iname)

            # Fixed values as constants
            for fname in force_spec.nonbonded_fixed_names:
                val = parameter_set.get_fixed(fname)
                try:
                    values[fname] = float(val)
                except Exception:
                    values[fname] = val.value_in_unit(kilojoule_per_mole)

            # Construct ordered argument list for lambda function
            args = build_lambda_args(force_spec.nonbonded_eval_params, values)
            contrib = deriv_func(*args)

            # Add the sum of the contribution to the gradient, update counts
            first_grad["fixed"][pname] += float(np.sum(contrib))
            counts["fixed"][pname] += len(r)

    # We calculated the sum, but gradient values need to be averages
    for pname in first_grad["pair"]:
        for key in first_grad["pair"][pname]:
            first_grad["pair"][pname][key] /= max(counts["pair"][pname][key], 1)

    for pname in first_grad["individual"]:
        for key in first_grad["individual"][pname]:
            first_grad["individual"][pname][key] /= max(counts["individual"][pname][key], 1)

    for pname in first_grad["fixed"]:
        first_grad["fixed"][pname] /= max(counts["fixed"][pname], 1)

    return first_grad

# Function to accumulate bonded gradients
def accumulate_bonded_general(positions, boxes, topology, parameter_set, force_spec, bead_types_by_index, 
                              frame_stride=1):
    """
    Generalized bonded gradient accumulation using force_spec's bonded_first_derivatives

    Returns
    -------
    first_grad : dict
        Gradient dictionary, gradient for each parameter
    """
    # Initialize gradient and counts as zeros
    first_grad = init_gradient_dict(parameter_set)
    counts = init_gradient_dict(parameter_set)

    # Get the bonded indices specifically, and get those radii
    bond_indices = get_bond_index_pairs(topology)
    samples = compute_sampled_radii(positions=positions, boxes=boxes, pair_indices=bond_indices, 
                                    bead_types_by_index=bead_types_by_index, frame_stride=frame_stride,
                                    cutoff_nm=None)

    # Extract important quantities from samples
    r_all = samples["r"]
    pair_keys_all = samples["pair_keys"]
    n_samples = samples["n_samples"]

    # If no bonds, return zero for gradient
    if n_samples == 0:
        return first_grad

    # Pairwise bonded parameters
    for pname in force_spec.bonded_pair_names:
        deriv_func = force_spec.bonded_first_derivatives[pname]

        # Get pair key and mask it out
        for pair_key in parameter_set.pair_parameters[pname].keys():
            mask = pair_key_mask(pair_keys_all, pair_key)
            if not np.any(mask):
                continue

            bt_i, bt_j = pair_key
            r = r_all[mask]

            values = {"r": r}

            # Add pair, individual, and fixed parameters as constants
            for qname in force_spec.bonded_pair_names:
                if parameter_set.has_pair(pair_key, qname):
                    values[qname] = parameter_set.get_pair(pair_key, qname)

            for iname in force_spec.bonded_individual_names:
                values[f"{iname}1"] = parameter_set.get_individual(bt_i, iname)
                values[f"{iname}2"] = parameter_set.get_individual(bt_j, iname)

            for fname in force_spec.bonded_fixed_names:
                val = parameter_set.get_fixed(fname)
                try:
                    values[fname] = float(val)
                except Exception:
                    values[fname] = val.value_in_unit(kilojoule_per_mole)

            # Create args list, calculate gradient
            args = build_lambda_args(force_spec.bonded_eval_params, values)
            contrib = deriv_func(*args)

            first_grad["pair"][pname][pair_key] += float(np.sum(contrib))
            counts["pair"][pname][pair_key] += len(r)

    # Individual bonded parameters
    for pname in force_spec.bonded_individual_names:
        deriv_func_1 = force_spec.bonded_first_derivatives[f"{pname}1"]
        deriv_func_2 = force_spec.bonded_first_derivatives[f"{pname}2"]

        unique_pair_keys = set(pair_keys_all.tolist())

        # Get pair key and mask it out
        for pair_key in unique_pair_keys:
            mask = pair_key_mask(pair_keys_all, pair_key)
            if not np.any(mask):
                continue

            bt_i, bt_j = pair_key
            r = r_all[mask]

            values = {"r": r}

            # Add pair, individual, and fixed parameters as constants
            for qname in force_spec.bonded_pair_names:
                if parameter_set.has_pair(pair_key, qname):
                    values[qname] = parameter_set.get_pair(pair_key, qname)

            for iname in force_spec.bonded_individual_names:
                values[f"{iname}1"] = parameter_set.get_individual(bt_i, iname)
                values[f"{iname}2"] = parameter_set.get_individual(bt_j, iname)

            for fname in force_spec.bonded_fixed_names:
                val = parameter_set.get_fixed(fname)
                try:
                    values[fname] = float(val)
                except Exception:
                    values[fname] = val.value_in_unit(kilojoule_per_mole)

            # Create args list, calculate gradient
            args = build_lambda_args(force_spec.bonded_eval_params, values)

            contrib_1 = deriv_func_1(*args)
            contrib_2 = deriv_func_2(*args)

            first_grad["individual"][pname][bt_i] += float(np.sum(contrib_1))
            first_grad["individual"][pname][bt_j] += float(np.sum(contrib_2))
            counts["individual"][pname][bt_i] += len(r)
            counts["individual"][pname][bt_j] += len(r)

    # Fixed bonded parameters
    for pname in force_spec.bonded_fixed_names:
        deriv_func = force_spec.bonded_first_derivatives[pname]

        unique_pair_keys = set(pair_keys_all.tolist())

        # Get pair key and mask it out
        for pair_key in unique_pair_keys:
            mask = pair_key_mask(pair_keys_all, pair_key)
            if not np.any(mask):
                continue

            bt_i, bt_j = pair_key
            r = r_all[mask]

            values = {"r": r}

            # Add pair, individual, and fixed parameters as constants
            for qname in force_spec.bonded_pair_names:
                if parameter_set.has_pair(pair_key, qname):
                    values[qname] = parameter_set.get_pair(pair_key, qname)

            for iname in force_spec.bonded_individual_names:
                values[f"{iname}1"] = parameter_set.get_individual(bt_i, iname)
                values[f"{iname}2"] = parameter_set.get_individual(bt_j, iname)

            for fname in force_spec.bonded_fixed_names:
                val = parameter_set.get_fixed(fname)
                try:
                    values[fname] = float(val)
                except Exception:
                    values[fname] = val.value_in_unit(kilojoule_per_mole)

            # Create args list, calculate gradient
            args = build_lambda_args(force_spec.bonded_eval_params, values)
            contrib = deriv_func(*args)

            first_grad["fixed"][pname] += float(np.sum(contrib))
            counts["fixed"][pname] += len(r)

    # Compute the average for all gradients
    for pname in first_grad["pair"]:
        for key in first_grad["pair"][pname]:
            first_grad["pair"][pname][key] /= max(counts["pair"][pname][key], 1)

    for pname in first_grad["individual"]:
        for key in first_grad["individual"][pname]:
            first_grad["individual"][pname][key] /= max(counts["individual"][pname][key], 1)

    for pname in first_grad["fixed"]:
        first_grad["fixed"][pname] /= max(counts["fixed"][pname], 1)

    return first_grad


# Construct gradient dictionary for srel
def calculate_srel_gradients(aa_positions, cg_positions, topology, parameter_set, force_spec, aa_boxes=None, 
                             cg_boxes=None, scale_by_beta=True, cutoff_nm=None, frame_stride=1):
    """
    Generalized srel gradient calculation

    Parameters
    ----------
    aa/cg_positions : np.ndarray
        Shape (n_frames, n_beads, 3), mapped AA->CG or CG positions in nm
    topology : openmm topology object
        CG topology, applies to both AA and CG mapped positions
    parameter_set : ParameterSet
        Current parameters for the gradient calculation
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
    aa_nb = accumulate_nonbonded_general(aa_positions, aa_boxes, topology, parameter_set, force_spec, bead_types_by_index, frame_stride, cutoff_nm)
    cg_nb = accumulate_nonbonded_general(cg_positions, cg_boxes, topology, parameter_set, force_spec, bead_types_by_index, frame_stride, cutoff_nm)

    # Accumulate bonded for AA and CG
    aa_bd = accumulate_bonded_general(aa_positions, aa_boxes, topology, parameter_set, force_spec, bead_types_by_index, frame_stride)
    cg_bd = accumulate_bonded_general(cg_positions, cg_boxes, topology, parameter_set, force_spec, bead_types_by_index, frame_stride)

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

def load_cg_trajectory_arrays(cg_topology_path, cg_trajectory_path, start=None, stop=None, step=None):
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

def map_aa_trajectory_to_cg_arrays(topology_path, trajectory_path, mapping_rules, start=None, stop=None, step=None):
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

                # Handle the openmm quantities gracefully
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

#################### Gradient Writing Utils ########################

# Write out the gradient dictionary
def write_gradient_dict_to_csv(gradients, output_dir):
    """
    Write gradient dictionary to 3 csv files:
        - pair_gradients.csv
        - individual_gradients.csv
        - fixed_gradients.csv

    Parameters
    ----------
    gradients : dict
        Format:
        {
            "pair": {pname: {pair_key: grad}},
            "individual": {pname: {bead_type: grad}},
            "fixed": {pname: grad},
        }
    output_dir : str
        Directory to write csv files into
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pairwise gradients
    pair_file = os.path.join(output_dir, "pair_gradients.csv")
    pair_gradients = gradients.get("pair", {})

    if pair_gradients:
        all_pairs = set()
        pair_names = sorted(pair_gradients.keys())

        for pname in pair_names:
            all_pairs.update(pair_gradients[pname].keys())

        all_pairs = sorted(all_pairs)

        with open(pair_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bead_type_i", "bead_type_j"] + pair_names)

            for pair in all_pairs:
                row = [pair[0], pair[1]]
                for pname in pair_names:
                    row.append(pair_gradients.get(pname, {}).get(pair, ""))
                writer.writerow(row)

    # Individual gradients
    indiv_file = os.path.join(output_dir, "individual_gradients.csv")
    indiv_gradients = gradients.get("individual", {})

    if indiv_gradients:
        all_bead_types = set()
        indiv_names = sorted(indiv_gradients.keys())

        for pname in indiv_names:
            all_bead_types.update(indiv_gradients[pname].keys())

        all_bead_types = sorted(all_bead_types)

        with open(indiv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bead_type"] + indiv_names)

            for bead_type in all_bead_types:
                row = [bead_type]
                for pname in indiv_names:
                    row.append(indiv_gradients.get(pname, {}).get(bead_type, ""))
                writer.writerow(row)

    # Fixed gradients
    fixed_file = os.path.join(output_dir, "fixed_gradients.csv")
    fixed_gradients = gradients.get("fixed", {})

    if fixed_gradients:
        with open(fixed_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            for pname, value in sorted(fixed_gradients.items()):
                writer.writerow([pname, value])

# Helper - write gradients out for a given step
def write_intermediate_gradients(gradients, project_name, iteration):
    """
    Write gradients for one iteration to:
        project_name/cg_gradients/iteration_<iteration>/
    """
    output_dir = os.path.join(project_name, "cg_gradients", f"iteration_{iteration}")
    write_gradient_dict_to_csv(gradients, output_dir)

