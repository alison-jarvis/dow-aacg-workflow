#!/bin/sh

sudo apptainer run --nvccli smiles.sif python src/run_minimization.py "$@"
