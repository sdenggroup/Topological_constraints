#!/bin/bash
#SBATCH -N 2
#SBATCH -n 112
#SBATCH -p cp4
#SBATCH -J shg_1d_alpha_lowmem
#SBATCH -o shg_1d_alpha_lowmem.%j.out
#SBATCH -e shg_1d_alpha_lowmem.%j.err
module purge
module load Intel_compiler/19.0.4
module load MKL/19.1.2
module load MPI/mpich/4.0.2-mpi-x-icc19.0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON_PATH=$HOME/miniconda3/envs/SHGmodel/bin/python
SCRIPT_NAME=1DSHGline.py
LOG_NAME=shg_1d_alpha_kparallel_lowmem_mpi_calc.log
yhrun -n ${SLURM_NTASKS:-224} $PYTHON_PATH -u $SCRIPT_NAME > $LOG_NAME 2>&1
RET=$?
exit $RET
