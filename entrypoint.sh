#!/bin/sh

set -e

umask 000
mkdir -p /repo
cd /repo
bash "$@"