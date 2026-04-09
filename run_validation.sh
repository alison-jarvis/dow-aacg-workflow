#!/bin/sh

sudo apptainer run --nvccli smiles.sif python validation/run_validation.py "$@"