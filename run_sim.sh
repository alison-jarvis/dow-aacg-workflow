#!/bin/sh

docker run --rm -it -v $(pwd):/repo capstone/smiles python src/run_aamd.py "$@"