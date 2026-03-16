#!/bin/sh

docker run --rm -it --user "$(id -u):$(id -g)" -v $(pwd):/repo capstone/smiles python src/run_aamd.py "$@"