#!/bin/sh
set -e

umask 000
mkdir -p /repo
cd /repo

# The micromamba entrypoint has already activated Python by the time we get here.
# Just execute the passed arguments natively.
exec "$@"