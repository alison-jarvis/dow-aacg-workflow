#!/bin/sh

docker run --rm -it --user "$(id -u):$(id -g)" --gpus all -v "$(pwd)":/repo capstone/smiles python src/run_aamd.py "$@"