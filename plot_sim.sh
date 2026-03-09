#!/bin/sh

docker run --rm -it -v $(pwd):/repo capstone/smiles python src/plot_aamd.py "$@"