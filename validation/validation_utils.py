import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
from pathlib import Path
import math

def aggregate_bead_data(project_name):
    folder_path = project_name + "/cg_parameters/"
    # Initialize the nested structure
    # { 'a': {}, 'gamma': {} }
    data_map = {'a': {}, 'gamma': {}}

    # Iterate through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    bead = row['bead_type']
                    gamma_val = float(row['gamma'])
                    a_val = float(row['a'])
                    
                    # Process Gamma
                    if bead not in data_map['gamma']:
                        data_map['gamma'][bead] = []
                    data_map['gamma'][bead].append(gamma_val)
                    
                    # Process A
                    if bead not in data_map['a']:
                        data_map['a'][bead] = []
                    data_map['a'][bead].append(a_val)
                    
    return data_map

def load_rdf_errors(project_name):
    rdf_folder = project_name + "/rdf_errors.csv"
    # Load in rdf errors
    rdf_errors = []
    with open(rdf_folder, mode='r', newline='') as file:
        reader = list(csv.DictReader(file))
        for val in reader:
            rdf_errors.append(float(val['RDF Error']))
    return rdf_errors


# Plot the coarse grain minimization evolution
def visualize_cg_minimization(project_name):
    # Load rdf errors and parameters
    parameter_dict = aggregate_bead_data(project_name)
    rdf_errors = load_rdf_errors(project_name)
    # Plot the rdf errors and parameter evolution
    fig, ax = plt.subplots(3, sharex=True, figsize=(10, 8))
    ax[0].plot(rdf_errors)
    ax[0].set_ylabel('RDF Errors')
    for bead_type in parameter_dict['a'].keys():
        ax[1].plot(parameter_dict['a'][bead_type], label=bead_type)
    ax[1].set_ylabel('Parameter A')
    ax[1].legend()
    for bead_type in parameter_dict['a'].keys():
        ax[2].plot(parameter_dict['gamma'][bead_type], label=bead_type)
    ax[2].set_ylabel('Parameter Gamma')
    ax[2].legend()
    ax[2].set_xlabel('Iteration Number')
    plt.suptitle('Srel Minimization Loop Evolution')
    fig.tight_layout()
    save_path = project_name + "/validation/cg_minimization.png"
    fig.savefig(save_path)


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
