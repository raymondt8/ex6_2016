#!/bin/bash

cd Build

module load mpt
module load intelcomp
module load cmake

CC=mpicc cmake .. -DCMAKE_BUILD_TYPE=Release
make

