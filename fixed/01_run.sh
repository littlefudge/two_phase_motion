#!/bin/bash
#PBS -d .
#PBS -N two_phase_motion
#PBS -j oe
#PBS -l nodes=10:ppn=12
#PBS -l walltime=24:00:00
set -euo pipefail

cd "$PBS_O_WORKDIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mpirun "$PYTHON_BIN" "$PBS_O_WORKDIR/main_test_rk45.py" --config "$PBS_O_WORKDIR/params.yaml"
