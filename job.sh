#!/bin/bash

# PBS -N poisson
# PBS -A ntnu603
# PBS -l walltime=00:01:00
# PBS -l select=2:ncpus=32:mpiprocs=16:ompthreads=2

cd $PBS_O_WORKDIR
cd Build
module load mpt
module load intelcomp

bash exercise2.sh

