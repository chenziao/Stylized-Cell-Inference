#!/usr/bin/env bash

runipy ../notebooks/GenerateGroundTruth.ipynb

python ../stylized_module/base/generate_priors.py
python ../stylized_module/base/simulate_cells.py
python ../stylized_module/base/inference.py
python ../stylized_module/base/generate_priors.py
python ../stylized_module/base/simulate_cells.py
python ../stylized_module/base/inference.py

runipy PlotInferenceResults.ipynb