# Imports
import csv
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import glob
import math


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

################ Plotting Utilities #####################

# Plot the coarse grain minimization evolution
def visualize_cg_minimization(project_name, include_fixed=False):
    """
    Plot RDF error and parameter evolution across minimization iterations.

    Expected directory structure:
        project_name/
            rdf_errors.csv
            cg_parameters/
                iteration_0/
                    pair_parameters.csv
                    individual_parameters.csv
                    fixed_parameters.csv
                iteration_1/
                    ...

    Output:
        project_name/validation/cg_minimization.png
    """

    histories, iterations = load_all_parameter_histories(project_name)
    rdf_errors = load_rdf_errors(project_name)

    pair_history = histories["pair"]
    individual_history = histories["individual"]
    fixed_history = histories["fixed"]

    # Build ordered list of subplot parameter groups
    plot_specs = [("rdf_error", None)]

    for pname in sorted(individual_history.keys()):
        plot_specs.append(("individual", pname))

    for pname in sorted(pair_history.keys()):
        plot_specs.append(("pair", pname))

    if include_fixed:
        for pname in sorted(fixed_history.keys()):
            plot_specs.append(("fixed", pname))

    n_plots = len(plot_specs)
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(12, 3.5 * n_plots))

    if n_plots == 1:
        axes = [axes]

    # RDF error
    axes[0].plot(range(len(rdf_errors)), rdf_errors)
    axes[0].set_ylabel("RDF Error")
    axes[0].set_title("RDF Error Evolution", loc="right")

    ax_idx = 1

    # Individual params
    for pname in sorted(individual_history.keys()):
        ax = axes[ax_idx]
        for bead_type, values in sorted(individual_history[pname].items()):
            ax.plot(iterations[:len(values)], values, label=bead_type)
        ax.set_ylabel(pname)
        ax.set_title(f"Individual Parameter: {pname}", loc="right")
        ax.legend(fontsize=8, ncol=2)
        ax_idx += 1

    # Pairwise params
    for pname in sorted(pair_history.keys()):
        ax = axes[ax_idx]
        for pair_id, values in sorted(pair_history[pname].items()):
            ax.plot(iterations[:len(values)], values, label=pair_id)
        ax.set_ylabel(pname)
        ax.set_title(f"Pairwise Parameter: {pname}", loc="right")
        ax.legend(fontsize=7, ncol=2)
        ax_idx += 1

    # Fixed params
    if include_fixed:
        for pname in sorted(fixed_history.keys()):
            ax = axes[ax_idx]
            values = fixed_history[pname]
            ax.plot(iterations[:len(values)], values, label=pname)
            ax.set_ylabel(pname)
            ax.set_title(f"Fixed Parameter: {pname}", loc="right")
            ax.legend(fontsize=8)
            ax_idx += 1

    axes[-1].set_xlabel("Iteration Number")
    plt.suptitle("CG Minimization Evolution")
    fig.tight_layout()

    save_dir = os.path.join(project_name, "validation")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "cg_minimization.png")
    fig.savefig(save_path)
    print(f"Saved minimization plot to {save_path}")

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

def plot_cgmd_diagnostics(project_name, n=70):
    # Folder where the log files exist
    cg_log_folder = project_name + "/cg_diagnostics/"

    # Create the plots
    plot_folder = project_name + "/validation/"

    ########## Energies ###########
    fig, ax = plt.subplots(3, sharex=True, figsize=(8, 6))
    bcmap = plt.get_cmap('Blues')
    rcmap = plt.get_cmap('Reds')
    gcmap = plt.get_cmap('Greens')
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

# Plot final rdfs
def plot_final_rdfs(project_name):
    # Load CSV data
    csv_path = project_name + "/final_rdfs.csv"
    df = pd.read_csv(csv_path)

    # Identify all prefixes ending with '_target' to find bead type combinations
    target_cols = [col for col in df.columns if col.endswith('_target')]
    prefixes = [col.replace('_target', '') for col in target_cols]

    # Create subplots grid dynamically based on the number of prefixes
    num_plots = len(prefixes)
    cols = 7
    rows = math.ceil(num_plots / cols)

    # Set up the figure
    fig, ax = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    ax = ax.flatten()

    # Plot the target and current data for each bead type
    for i, prefix in enumerate(prefixes):
        target_col = f"{prefix}_target"
        current_col = f"{prefix}_current"
        
        # Check if columns exist before plotting
        if target_col in df.columns and current_col in df.columns:
            ax[i].plot(df['r_nm'], df[target_col], label='AAMD (Target)', alpha=0.9)
            ax[i].plot(df['r_nm'], df[current_col], label='CGMD', alpha=0.9)
            
            ax[i].set_title(prefix, fontsize=10)
            ax[i].set_xlabel('Radius of Separation (nm)', fontsize=8)
            ax[i].set_ylabel('Distribution', fontsize=8)
            ax[i].legend(fontsize=8)
            ax[i].tick_params(axis='both', which='major', labelsize=8)

    # Hide any extra empty subplots if the grid is larger than the number of plots
    for j in range(len(prefixes), len(ax)):
        fig.delaxes(ax[j])

    # Save the plot
    plt.tight_layout()
    save_path = project_name + "/validation/final_rdfs.png"
    fig.savefig(save_path)
