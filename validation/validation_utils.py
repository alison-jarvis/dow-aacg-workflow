# Imports
import csv
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import glob
import math
import numpy as np


############ Csv parsing utilities #############

def extract_iteration_number(folder_name):
    """
    Extract integer iteration number from folder names like:
        iteration_0
        iteration_12
    """
    m = re.search(r"iteration_(\d+)", folder_name)
    if m is None:
        return None
    return int(m.group(1))


def discover_iteration_dirs(base_dir):
    """
    Generic helper for folders like:
        base_dir/iteration_0
        base_dir/iteration_1
        ...
    """
    if not os.path.isdir(base_dir):
        return []

    dirs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            iteration = extract_iteration_number(name)
            if iteration is not None:
                dirs.append((iteration, full))

    dirs.sort(key=lambda x: x[0])
    return dirs


def load_pair_gradient_history(project_name):
    base_dir = os.path.join(project_name, "cg_gradients")
    dirs = discover_iteration_dirs(base_dir)
    iterations = [it for it, _ in dirs]

    history = {}

    for _, folder in dirs:
        csv_path = os.path.join(folder, "pair_gradients.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        param_cols = [c for c in df.columns if c not in ["bead_type_i", "bead_type_j"]]

        for _, row in df.iterrows():
            pair_id = f"{row['bead_type_i']}--{row['bead_type_j']}"
            for pname in param_cols:
                history.setdefault(pname, {})
                history[pname].setdefault(pair_id, [])
                history[pname][pair_id].append(row[pname])

    return history, iterations


def load_individual_gradient_history(project_name):
    base_dir = os.path.join(project_name, "cg_gradients")
    dirs = discover_iteration_dirs(base_dir)
    iterations = [it for it, _ in dirs]

    history = {}

    for _, folder in dirs:
        csv_path = os.path.join(folder, "individual_gradients.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        param_cols = [c for c in df.columns if c != "bead_type"]

        for _, row in df.iterrows():
            bead_type = row["bead_type"]
            for pname in param_cols:
                history.setdefault(pname, {})
                history[pname].setdefault(bead_type, [])
                history[pname][bead_type].append(row[pname])

    return history, iterations


def load_fixed_gradient_history(project_name):
    base_dir = os.path.join(project_name, "cg_gradients")
    dirs = discover_iteration_dirs(base_dir)
    iterations = [it for it, _ in dirs]

    history = {}

    for _, folder in dirs:
        csv_path = os.path.join(folder, "fixed_gradients.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            pname = row["parameter"]
            history.setdefault(pname, [])
            history[pname].append(row["value"])

    return history, iterations


def load_all_gradient_histories(project_name):
    pair_history, pair_iters = load_pair_gradient_history(project_name)
    indiv_history, indiv_iters = load_individual_gradient_history(project_name)
    fixed_history, fixed_iters = load_fixed_gradient_history(project_name)

    iterations = pair_iters or indiv_iters or fixed_iters

    return {
        "pair": pair_history,
        "individual": indiv_history,
        "fixed": fixed_history,
    }, iterations


def find_parameter_iteration_dirs(project_name):
    """
    Return sorted list of iteration directories inside:
        project_name/cg_parameters/
    """
    base_dir = os.path.join(project_name, "cg_parameters")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Parameter directory not found: {base_dir}")

    dirs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            iteration = extract_iteration_number(name)
            if iteration is not None:
                dirs.append((iteration, full))

    dirs.sort(key=lambda x: x[0])
    return dirs


def load_pair_parameter_history(project_name):
    """
    Load pairwise parameter history from:
        cg_parameters/iteration_X/pair_parameters.csv

    Returns
    -------
    pair_history : dict
        {
            "pair_param": {
                "TYPE1--TYPE2": [val_iter0, val_iter1, ...],
                ...
            },
            ...
        }
    iterations : list[int]
    """
    dirs = find_parameter_iteration_dirs(project_name)
    iterations = [it for it, _ in dirs]

    pair_history = {}

    for _, folder in dirs:
        csv_path = os.path.join(folder, "pair_parameters.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # parameter columns are everything except the identifier columns
        param_cols = [c for c in df.columns if c not in ["bead_type_i", "bead_type_j"]]

        for _, row in df.iterrows():
            pair_id = f"{row['bead_type_i']}--{row['bead_type_j']}"

            for pname in param_cols:
                pair_history.setdefault(pname, {})
                pair_history[pname].setdefault(pair_id, [])
                pair_history[pname][pair_id].append(row[pname])

    return pair_history, iterations


def load_individual_parameter_history(project_name):
    """
    Load individual parameter history from:
        cg_parameters/iteration_X/individual_parameters.csv

    Returns
    -------
    individual_history : dict
        {
            "indiv_param": {
                "WAT_W": [val_iter0, val_iter1, ...],
                ...
            }
        }
    iterations : list[int]
    """
    dirs = find_parameter_iteration_dirs(project_name)
    iterations = [it for it, _ in dirs]

    individual_history = {}

    for _, folder in dirs:
        csv_path = os.path.join(folder, "individual_parameters.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        param_cols = [c for c in df.columns if c != "bead_type"]

        for _, row in df.iterrows():
            bead_type = row["bead_type"]

            for pname in param_cols:
                individual_history.setdefault(pname, {})
                individual_history[pname].setdefault(bead_type, [])
                individual_history[pname][bead_type].append(row[pname])

    return individual_history, iterations


def load_fixed_parameter_history(project_name):
    """
    Load fixed parameter history from:
        cg_parameters/iteration_X/fixed_parameters.csv

    Returns
    -------
    fixed_history : dict
        {
            "fixed_param": [val_iter0, val_iter1, ...],
            ...
        }
    iterations : list[int]
    """
    dirs = find_parameter_iteration_dirs(project_name)
    iterations = [it for it, _ in dirs]

    fixed_history = {}

    for _, folder in dirs:
        csv_path = os.path.join(folder, "fixed_parameters.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            pname = row["parameter"]
            val = row["value"]

            fixed_history.setdefault(pname, [])
            fixed_history[pname].append(val)

    return fixed_history, iterations


def load_all_parameter_histories(project_name):
    """
    Wrapper, for convenience

    Returns
    -------
    histories : dict
        {
            "pair": ...,
            "individual": ...,
            "fixed": ...
        }
    iterations : list[int]
    """
    # Loads the histories from all types from project
    pair_history, pair_iters = load_pair_parameter_history(project_name)
    indiv_history, indiv_iters = load_individual_parameter_history(project_name)
    fixed_history, fixed_iters = load_fixed_parameter_history(project_name)

    # Use whichever iteration array is non-empty / longest
    iterations = pair_iters or indiv_iters or fixed_iters

    histories = {"pair": pair_history, "individual": indiv_history, "fixed": fixed_history}

    return histories, iterations


# Load the rdf errors from the csv
def load_rdf_errors(project_name):
    rdf_folder = project_name + "/rdf_errors.csv"
    # Load in rdf errors
    rdf_errors = []
    with open(rdf_folder, mode='r', newline='') as file:
        reader = list(csv.DictReader(file))
        for val in reader:
            rdf_errors.append(float(val['RDF Error']))
    return rdf_errors


def compute_gradient_norms(gradient_histories, iterations):
    """
    Compute L2 norm of all available gradient values at each iteration

    Parameters
    ----------
    gradient_histories : dict
        Output of load_all_gradient_histories(project_name)
    iterations : list[int]

    Returns
    -------
    norms : list[float]
    """
    n_iters = len(iterations)
    norms = []

    for i in range(n_iters):
        vals = []

        # Pair gradients
        for pname, series_dict in gradient_histories.get("pair", {}).items():
            for _, series in series_dict.items():
                if series is None or len(series) == 0:
                    continue
                if i < len(series):
                    val = series[i]
                    if pd.notna(val):
                        vals.append(float(val))

        # Individual gradients
        for pname, series_dict in gradient_histories.get("individual", {}).items():
            for _, series in series_dict.items():
                if series is None or len(series) == 0:
                    continue
                if i < len(series):
                    val = series[i]
                    if pd.notna(val):
                        vals.append(float(val))

        # Fixed gradients
        for pname, series in gradient_histories.get("fixed", {}).items():
            if series is None or len(series) == 0:
                continue
            if i < len(series):
                val = series[i]
                if pd.notna(val):
                    vals.append(float(val))

        if len(vals) == 0:
            norms.append(0.0)
        else:
            norms.append(float(np.linalg.norm(vals)))

    return norms

def series_has_data(series):
    """
    True if series contains at least one valid numeric value.
    """
    if series is None or len(series) == 0:
        return False

    for v in series:
        if pd.notna(v):
            return True
    return False


def dict_of_series_has_data(series_dict):
    """
    True if at least one series in the dict has plottable data
    """
    if series_dict is None or len(series_dict) == 0:
        return False

    for _, series in series_dict.items():
        if series_has_data(series):
            return True
    return False


def filter_history_group(history_group):
    """
    Remove parameter entries with no plottable data

    Parameters
    ----------
    history_group : dict
        e.g. history_group[pname][series_name] = list_of_values

    Returns
    -------
    filtered : dict
    """
    filtered = {}

    for pname, series_dict in history_group.items():
        if dict_of_series_has_data(series_dict):
            filtered[pname] = {sname: series for sname, series in series_dict.items() if series_has_data(series)}

    return filtered


def filter_fixed_history(history_group):
    """
    Remove fixed-parameter entries with no plottable data
    """
    filtered = {}
    for pname, series in history_group.items():
        if series_has_data(series):
            filtered[pname] = series
    return filtered

################ Plotting Utilities #####################

# Visualize the minimization, parameters and gradient
def visualize_cg_minimization(project_name, include_fixed=False):
    """
    Plot gradient evolution and parameter evolution across minimization iterations

    Only includes subplots for parameters that actually contain plottable data
    """

    # Load parameter histories
    param_histories, param_iterations = load_all_parameter_histories(project_name)
    pair_history = filter_history_group(param_histories.get("pair", {}))
    individual_history = filter_history_group(param_histories.get("individual", {}))
    fixed_history = filter_fixed_history(param_histories.get("fixed", {}))

    # Load gradient histories
    grad_histories, grad_iterations = load_all_gradient_histories(project_name)
    grad_pair_history = filter_history_group(grad_histories.get("pair", {}))
    grad_individual_history = filter_history_group(grad_histories.get("individual", {}))
    grad_fixed_history = filter_fixed_history(grad_histories.get("fixed", {}))

    filtered_grad_histories = {"pair": grad_pair_history, 
                               "individual": grad_individual_history,
                               "fixed": grad_fixed_history}

    gradient_norms = compute_gradient_norms(filtered_grad_histories, grad_iterations)

    # Determine whether gradient subplot has anything to show
    has_gradient_data = (
        len(grad_pair_history) > 0
        or len(grad_individual_history) > 0
        or (include_fixed and len(grad_fixed_history) > 0)
        or any(abs(x) > 0 for x in gradient_norms)
    )

    # Build ordered list of subplot parameter groups
    plot_specs = []

    if has_gradient_data:
        plot_specs.append(("gradient", None))

    for pname in sorted(individual_history.keys()):
        plot_specs.append(("individual", pname))

    for pname in sorted(pair_history.keys()):
        plot_specs.append(("pair", pname))

    if include_fixed:
        for pname in sorted(fixed_history.keys()):
            plot_specs.append(("fixed", pname))

    # Nothing to plot
    if len(plot_specs) == 0:
        print(f"No minimization history data found to plot for project {project_name}.")
        return

    n_plots = len(plot_specs)
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(12, 3.8 * n_plots))

    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    # Gradient subplot
    if has_gradient_data:
        ax0 = axes[ax_idx]

        # Pair gradients
        for pname in sorted(grad_pair_history.keys()):
            for pair_id, values in sorted(grad_pair_history[pname].items()):
                if not series_has_data(values):
                    continue
                ax0.plot(
                    grad_iterations[:len(values)],
                    values,
                    label=f"{pname}: {pair_id}",
                    alpha=0.8
                )

        # Individual gradients
        for pname in sorted(grad_individual_history.keys()):
            for bead_type, values in sorted(grad_individual_history[pname].items()):
                if not series_has_data(values):
                    continue
                ax0.plot(grad_iterations[:len(values)], values, 
                         label=f"{pname}: {bead_type}", alpha=0.8, linestyle="--")

        # Fixed gradients
        if include_fixed:
            for pname, values in sorted(grad_fixed_history.items()):
                if not series_has_data(values):
                    continue
                ax0.plot(grad_iterations[:len(values)], values, label=f"{pname}", 
                         alpha=0.8, linestyle=":")

        # Norm
        if len(gradient_norms) > 0:
            ax0.plot(grad_iterations[:len(gradient_norms)], gradient_norms, 
                     label="norm", linewidth=2.5)

        ax0.set_ylabel("Gradient")
        ax0.set_title("Gradient Evolution", loc="right")
        ax0.legend(fontsize=7, ncol=2)
        ax_idx += 1

    # Individual parameter plots
    for pname in sorted(individual_history.keys()):
        if pname not in individual_history or len(individual_history[pname]) == 0:
            continue

        ax = axes[ax_idx]
        plotted_any = False

        for bead_type, values in sorted(individual_history[pname].items()):
            if not series_has_data(values):
                continue
            ax.plot(param_iterations[:len(values)], values, label=bead_type)
            plotted_any = True

        if plotted_any:
            ax.set_ylabel(pname)
            ax.set_title(f"Individual Parameter: {pname}", loc="right")
            ax.legend(fontsize=8, ncol=2)
            ax_idx += 1

    # Pairwise parameter plots
    for pname in sorted(pair_history.keys()):
        if pname not in pair_history or len(pair_history[pname]) == 0:
            continue

        ax = axes[ax_idx]
        plotted_any = False

        for pair_id, values in sorted(pair_history[pname].items()):
            if not series_has_data(values):
                continue
            ax.plot(param_iterations[:len(values)], values, label=pair_id)
            plotted_any = True

        if plotted_any:
            ax.set_ylabel(pname)
            ax.set_title(f"Pairwise Parameter: {pname}", loc="right")
            ax.legend(fontsize=7, ncol=2)
            ax_idx += 1

    # Fixed parameter plots
    if include_fixed:
        for pname in sorted(fixed_history.keys()):
            values = fixed_history[pname]
            if not series_has_data(values):
                continue

            ax = axes[ax_idx]
            ax.plot(param_iterations[:len(values)], values, label=pname)
            ax.set_ylabel(pname)
            ax.set_title(f"Fixed Parameter: {pname}", loc="right")
            ax.legend(fontsize=8)
            ax_idx += 1

    # If some empty plots were skipped, trim unused axes
    for j in range(ax_idx, len(axes)):
        fig.delaxes(axes[j])

    # Remaining axes after deletion
    remaining_axes = fig.axes
    if len(remaining_axes) > 0:
        remaining_axes[-1].set_xlabel("Iteration Number")

    plt.suptitle("CG Minimization Evolution")
    fig.tight_layout()

    save_dir = os.path.join(project_name, "validation")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "cg_minimization.png")
    fig.savefig(save_path)
    print(f"Saved minimization plot to {save_path}")


# All atom diagnostic plots
def plot_aamd_diagnostics(project_name):
    # Find all atom log path
    log_format = project_name + "/*_log.txt"
    aa_logs = glob.glob(log_format)
    log_path = aa_logs[0]

    # Load in the log file as a pandas df
    df = pd.read_csv(log_path)

    # Create the plots
    plot_folder = project_name + "/validation/"

    ########## Energies ###########
    fig, ax = plt.subplots(3, sharex=True, figsize=(8, 6))
    ax[0].plot(df['#"Step"'], df['Potential Energy (kJ/mole)'])
    ax[0].set_title('Potential Energy', loc='right')
    ax[0].set_ylabel('Energy (kJ/mol)')
    ax[1].plot(df['#"Step"'], df['Kinetic Energy (kJ/mole)'])
    ax[1].set_title('Kinetic Energy', loc='right')
    ax[1].set_ylabel('Energy (kJ/mol)')
    ax[2].plot(df['#"Step"'], df['Total Energy (kJ/mole)'])
    ax[2].set_title('Total Energy', loc='right')
    ax[2].set_ylabel('Energy (kJ/mol)')
    ax[2].set_xlabel('Simulation Step')
    plt.suptitle('All-Atom Simulation Energies')
    fig.tight_layout()
    fig.savefig(plot_folder + 'aa_energy.png')

    ########### Temperature #############
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df['#"Step"'], df["Temperature (K)"])
    plt.suptitle('All-Atom Simulation Temperature')
    ax.set_ylabel('Temperature (K)')
    ax.set_xlabel('Simulation Step')
    fig.tight_layout()
    fig.savefig(plot_folder + 'aa_temperature.png')

    ########### Density #############
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(df['#"Step"'], df["Density (g/mL)"])
    plt.suptitle('All-Atom Simulation Density')
    ax.set_ylabel('Density (g/mL)')
    ax.set_xlabel('Simulation Step')
    fig.tight_layout()
    fig.savefig(plot_folder + 'aa_density.png')

    ########### Volume #############
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df['#"Step"'], df["Box Volume (nm^3)"])
    plt.suptitle('All-Atom Simulation Volume')
    ax.set_ylabel('Volume ($nm^3$)')
    ax.set_xlabel('Simulation Step')
    fig.tight_layout()
    fig.savefig(plot_folder + 'aa_volume.png')

def plot_cgmd_diagnostics(project_name, margin=20):
    # Folder where the log files exist
    cg_log_folder = project_name + "/cg_diagnostics/"

    # Create the plots
    plot_folder = project_name + "/validation/"

    ########## Energies ###########
    fig, ax = plt.subplots(3, sharex=True, figsize=(8, 6))
    bcmap = plt.get_cmap('Blues')
    rcmap = plt.get_cmap('Reds')
    gcmap = plt.get_cmap('Greens')
    n = int(len(os.listdir(cg_log_folder)) * (1 + margin/100))
    # Iterate through logs
    for i, file in enumerate(sorted(os.listdir(cg_log_folder))):
        log_path = os.path.join(cg_log_folder, file)
        df = pd.read_csv(log_path)
        ax[0].plot(df['#"Step"'], df['Potential Energy (kJ/mole)'],  color = bcmap(1 - (i/n)), alpha=0.8)
        ax[1].plot(df['#"Step"'], df['Kinetic Energy (kJ/mole)'], color = rcmap(1 - (i/n)), alpha=0.8)
        ax[2].plot(df['#"Step"'], df['Total Energy (kJ/mole)'], color = gcmap(1 - (i/n)), alpha=0.8)
    ax[0].set_title('Potential Energy', loc='right')
    ax[0].set_ylabel('Energy (kJ/mol)')
    ax[1].set_title('Kinetic Energy', loc='right')
    ax[1].set_ylabel('Energy (kJ/mol)')
    ax[2].set_title('Total Energy', loc='right')
    ax[2].set_ylabel('Energy (kJ/mol)')
    ax[2].set_xlabel('Simulation Step')
    plt.suptitle('Coarse-Grained Simulation Energies')
    fig.tight_layout()
    fig.savefig(plot_folder + 'cg_energy.png')

    ########### Temperature #############
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('Blues')
    for i, file in enumerate(sorted(os.listdir(cg_log_folder))):
        log_path = os.path.join(cg_log_folder, file)
        df = pd.read_csv(log_path)
        ax.plot(df['#"Step"'], df["Temperature (K)"], color = cmap(1 - (i/n)), alpha=0.8)
    plt.suptitle('Coarse-Grained Simulation Temperature')
    ax.set_ylabel('Temperature (K)')
    ax.set_xlabel('Simulation Step')
    fig.tight_layout()
    fig.savefig(plot_folder + 'cg_temperature.png')

    ########### Density #############
    fig, ax = plt.subplots(figsize=(7,5))
    cmap = plt.get_cmap('Blues')
    for i, file in enumerate(sorted(os.listdir(cg_log_folder))):
        log_path = os.path.join(cg_log_folder, file)
        df = pd.read_csv(log_path)
        ax.plot(df['#"Step"'], df["Density (g/mL)"], color = cmap(1 - (i/n)), alpha=0.8)
    plt.suptitle('Coarse-Grained Simulation Density')
    ax.set_ylabel('Density (g/mL)')
    ax.set_xlabel('Simulation Step')
    fig.tight_layout()
    fig.savefig(plot_folder + 'cg_density.png')

    ########### Volume #############
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('Blues')
    for i, file in enumerate(sorted(os.listdir(cg_log_folder))):
        log_path = os.path.join(cg_log_folder, file)
        df = pd.read_csv(log_path)
        ax.plot(df['#"Step"'], df["Box Volume (nm^3)"], color = cmap(1 - (i/n)), alpha=0.8)
    plt.suptitle('Coarse-Grained Simulation Volume')
    ax.set_ylabel('Volume ($nm^3$)')
    ax.set_xlabel('Simulation Step')
    fig.tight_layout()
    fig.savefig(plot_folder + 'cg_volume.png')

# Plot the radial distribution function
def plot_final_rdfs(project_name):

    paths = {"final_rdfs.csv":"final_rdfs.png", "starting_rdfs.csv":"starting_rdfs.png"}

    for csv_path_key in paths.keys():
        csv_path = os.path.join(project_name, csv_path_key)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find RDF file: {csv_path}")

        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"RDF file is empty: {csv_path}")
            return

        # Find target/current RDF columns
        target_cols = [col for col in df.columns if col.endswith("_target")]
        prefixes = [col[:-7] for col in target_cols]  # remove "_target"

        if len(prefixes) == 0:
            print(f"No RDF pair columns found in {csv_path}.")
            print(f"Columns present: {list(df.columns)}")
            return

        # Identify radius column
        if "r_nm" in df.columns:
            r_col = "r_nm"
        elif "r" in df.columns:
            r_col = "r"
        else:
            r_col = df.columns[0]

        num_plots = len(prefixes)
        cols = min(7, num_plots)
        rows = math.ceil(num_plots / cols)

        fig, ax = plt.subplots(rows, cols, figsize=(20, 4 * rows))

        # Normalize ax to a flat list
        if num_plots == 1:
            ax = [ax]
        else:
            ax = ax.flatten()

        for i, prefix in enumerate(prefixes):
            target_col = f"{prefix}_target"
            current_col = f"{prefix}_current"

            if target_col not in df.columns or current_col not in df.columns:
                continue

            ax[i].plot(df[r_col], df[target_col], label="AAMD (Target)", alpha=0.9)
            ax[i].plot(df[r_col], df[current_col], label="CGMD", alpha=0.9)

            ax[i].set_title(prefix, fontsize=10)
            ax[i].set_xlabel("Radius of Separation (nm)", fontsize=8)
            ax[i].set_ylabel("Distribution", fontsize=8)
            ax[i].legend(fontsize=8)
            ax[i].tick_params(axis="both", which="major", labelsize=8)

        # Remove unused axes
        for j in range(len(prefixes), len(ax)):
            fig.delaxes(ax[j])

        plt.tight_layout()

        save_dir = os.path.join(project_name, "validation")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, paths[csv_path_key])
        fig.savefig(save_path)
        print(f"Saved final RDF plot to {save_path}")