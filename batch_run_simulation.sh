#!/bin/bash

#SBATCH -J run_simulation
#SBATCH -o ./stdout/run_simulation_%j.out
#SBATCH -e ./stdout/run_simulation_%j.error
#SBATCH -t 0-48:00:00

#SBATCH -N 1
#SBATCH -n 1
##SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G


START=$(date)

unset DISPLAY
python data_simulation_loc_trunk_length.py

END=$(date)

echo "Started running at $START."
echo "Done running simulation at $END"