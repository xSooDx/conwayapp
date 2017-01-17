#!/bin/bash

module load libraries/openmpi-2.0.1-gcc-5.4.0
module load compilers/gnu-5.4.0
module load libraries/cuda

make
mpirun -np 5 ./game_of_life
