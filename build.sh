#!/bin/sh

# 1. Detect CUDA directly from the host terminal
HOST_CUDA=$(nvidia-smi | grep -E -o "CUDA Version: [0-9]+\.[0-9]+" | awk '{print $3}')

# Fallback just in case nvidia-smi fails or you build on a CPU-only login node
if [ -z "$HOST_CUDA" ]; then
    echo "Warning: Could not detect CUDA version"
    HOST_CUDA=""
else
    HOST_CUDA="11.8"
    echo "Force version: $HOST_CUDA"
fi

chmod +x ./entrypoint.sh

# Force the use of 11.8 or None
sudo apptainer build --build-arg CUDA_VER=$HOST_CUDA smiles.sif mamba_env.def