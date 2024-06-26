#!/bin/bash

#SBATCH -J run_multi_simulation_biophys
#SBATCH -o ./stdout/run_multi_simulation_biophys.%A_%a.out
#SBATCH -e ./stdout/run_multi_simulation_biophys.%A_%a.error
#SBATCH -t 0-48:00:00

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -a 0-74%25 # job array, % limiting the number of tasks that run at once
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G # suggest 16G for -cpus-per-task no more than 50


START=$(date)

unset DISPLAY
python data_simulation_biophysics.py ${SLURM_ARRAY_TASK_ID} -c 80 -l 1

END=$(date)

echo "Started running at $START."
echo "Done running simulation at $END"