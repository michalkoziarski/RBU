#!/bin/bash

#SBATCH -A imba
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --time=72:00:00

module add plgrid/tools/python/3.6.5

python3 -W ignore ${1} ${@:2}
