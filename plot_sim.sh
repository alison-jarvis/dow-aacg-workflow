#!/bin/sh

sudo apptainer run --nvccli smiles.sif python src/plot_aamd.py "$@"