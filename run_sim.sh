#!/bin/sh

# Runs your python simulation with GPU support.
# Replace 'run_simulation.py' with your actual python script.
# "$@" passes any CLI flags right through to python.
apptainer run --nv smiles.sif python src/run_aamd.py "$@"