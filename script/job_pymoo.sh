#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -l walltime=72:00:00

cd "$PBS_O_WORKDIR"
python3 test_pymoo_torque.py "$@"