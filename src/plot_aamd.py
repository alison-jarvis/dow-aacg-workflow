from aamd_utils import plot_diagnostics
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to plot AAMD quantities given a project name.")
    parser.add_argument("project", type=str, help="The name of the project to create plots for.")
    args = parser.parse_args()
    project_name = args.project
    p = Path(project_name)
    if not p.is_dir():
        raise Exception(f"{project_name} does not exist.")
    log_files = list(p.glob('*log.txt'))
    if not log_files:
        raise Exception(f"No log files exist for {project_name}")
    else:
        log_path = str(log_files[0])
        plot_diagnostics(log_path, project_name)
