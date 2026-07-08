#!/bin/bash
#SBATCH -N 2
#SBATCH -n 112
#SBATCH -p cp6

module purge
module load Intel_compiler/19.0.4
module load MKL/19.1.2
module load MPI/mpich/4.0.2-mpi-x-icc19.0

PYTHON_PATH=$HOME/miniconda3/envs/SHGmodel/bin/python
SCRIPT_NAME=2DSHG.py

yhrun $PYTHON_PATH -u $SCRIPT_NAME > shg_calc.log 2>&1
