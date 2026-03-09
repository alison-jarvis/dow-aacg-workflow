#!/bin/sh

docker run --rm -it --device nvidia.com/gpu=all -v $(pwd):/repo capstone/smiles