#!/bin/sh

chmod +x ./entrypoint.sh

# Builds the Apptainer image. (Requires sudo on most systems)
sudo apptainer build smiles.sif mamba_env.def