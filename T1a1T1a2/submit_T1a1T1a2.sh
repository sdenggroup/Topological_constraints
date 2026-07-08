#!/bin/bash
#SBATCH -N 4
#SBATCH -n 224
#SBATCH -p cp6
#SBATCH -J phase_shg_reflect_color
#SBATCH -o phase_shg_reflect_color.%j.out
#SBATCH -e phase_shg_reflect_color.%j.err

module purge
module load Intel_compiler/19.0.4
module load MKL/19.1.2
module load MPI/mpich/4.0.2-mpi-x-icc19.0

PYTHON_PATH=$HOME/miniconda3/envs/SHGmodel/bin/python
SCRIPT_NAME=T1a1T1a2.py

yhrun $PYTHON_PATH -u $SCRIPT_NAME
