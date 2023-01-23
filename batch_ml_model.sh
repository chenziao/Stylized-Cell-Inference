#!/bin/bash

#SBATCH -J ml_model
#SBATCH -o ./stdout/ml_model%j.out
#SBATCH -e ./stdout/ml_model%j.error
#SBATCH -t 0-48:00:00

#SBATCH -N 1
#SBATCH -n 1
##SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G


START=$(date)

python stylized_cell_NN_train.py -trial Reduced_Order_stochastic_lognG_spkwid_trunkLR4_LactvCa_Loc3_h1 -e 100 --train --cnn

END=$(date)

echo "Started running at $START."
echo "Done running model at $END"