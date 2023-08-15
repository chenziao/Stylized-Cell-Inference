#!/bin/bash

#SBATCH -J run_multi_simulation_pred
#SBATCH -o ./stdout/run_multi_simulation_pred.%A_%a.out
#SBATCH -e ./stdout/run_multi_simulation_pred.%A_%a.error
#SBATCH -t 0-48:00:00

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -a 0-19%30 # job array, % limiting the number of tasks that run at once
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G # suggest 16G for -c no more than 50


START=$(date)

unset DISPLAY
python data_simulation_prediction.py ${SLURM_ARRAY_TASK_ID} -c 10 -trial Reduced_Order_stochastic_spkwid_trunkLR4_LactvCa_Loc3_h1_sumstats -model FCN_batch256 -invivo all_cell_LFP2D_Analysis_SensorimotorSpikeWaveforms_NP_SUTempFilter_NPExample_v2 --stats

END=$(date)

echo "Started running at $START."
echo "Done running simulation at $END"