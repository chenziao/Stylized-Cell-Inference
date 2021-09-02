import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

#Dependencies
import numpy as np
import sbi.utils as utils
import torch

#Project Imports
import config.paths as paths
from stylized_module.models.cnn import SummaryNet


ACTIVE_CELL = True

#GENERAL PARAMETERS USED ACROSS RUNS
# ELECTRODE_POSITION = np.column_stack((np.zeros(96),np.linspace(-1900,1900,96),np.zeros(96)))
hf = h5py.File(paths.ELECTRODES,'r')
elec_pos = np.array(hf.get('coord'))
ELECTRODE_POSITION = np.column_stack((elec_pos,np.zeros(elec_pos.shape[0])))
ELECTRODE_GRID = (np.array(hf.get('grid/x')),np.array(hf.get('grid/y')),np.zeros(1))



#GET GROUND TRUTH FROM ACTIVE MODEL PARAMS ARE DENOTED WITH THE PREFIX GT
GT_TSTOP = 20. # ms
GT_DT = 0.025  # ms. does not allow change
GT_ELECTRODE_POSITION = np.column_stack((np.zeros(96),np.linspace(-1900,1900,96),np.zeros(96)))
GT_LOCATION_PARAMETERS = [0,350,80,1.43,0.9,1.27]
GT_GMAX = 0.0025
GT_SCALE = 1.
GT_BUTTERWORTH_ORDER = 2 #2nd order
GT_CRITICAL_FREQUENCY = 100 #100 Hz
GT_BANDFILTER_TYPE = 'hp' #highpass
GT_FILTER_SAMPLING_RATE = 40000 #40 kHz


#SIMULATION OF PASSIVE MODEL PARAMS ARE DENOTED WITH THE PREFIX PM
PM_TSTOP = 5. # ms
PM_DT = 0.025 # ms. does not allow change
PM_ELECTRODE_POSITION = np.column_stack((np.zeros(96),np.linspace(-1900,1900,96),np.zeros(96)))
PM_START_IDX = 360
PM_WINDOW_SIZE = 96
PM_SCALING_FACTOR = 1085. #2.55 


#SIMULATION OF ACTIVE MODEL PARAMS ARE DENOTED WITH THE PREFIX AM
AM_TSTOP = 20.
AM_DT = 0.025
AM_ELECTRODE_POSITION = np.column_stack((np.zeros(96),np.linspace(-1900,1900,96),np.zeros(96)))
AM_START_IDX = 390
AM_WINDOW_SIZE = 96
AM_TRUE_LOCATION_PARAMETERS = [0,350,80,3.0,0.9,1.27]
AM_FIXED_LOCATION_PARAMETERS = [0,380,75,2.9,0.92,1.2]


#INFERENCE MODEL PARAMS ARE DENOTED WITH THE PREFIX IM
IM_BUTTERWORTH_ORDER = 2 #2nd order
IM_CRITICAL_FREQUENCY = 100 #100 Hz
IM_BANDFILTER_TYPE = 'hp' #highpass
IM_FILTER_SAMPLING_RATE = 40000 #40 kHz
IM_Y_DISTANCE = GT_ELECTRODE_POSITION[:,1].ravel()
IM_EMBEDDED_NETWORK = SummaryNet(IM_Y_DISTANCE.size, PM_WINDOW_SIZE)
IM_THETA_BOUNDS = [0,np.pi]
#                       y          z            h       phi     sr      trl     trr         dr      tur         dl
IM_PARAMETER_BOUNDS = [[-2000,2000],[20,200],[-1,1],[0,np.pi],[3,12],[20,800],[0.2,1.0],[0.2,1.0],[0.2,1.0],[100,300]]#,[3,5]]
IM_PARAMETER_LOWS = torch.tensor([b[0] for b in IM_PARAMETER_BOUNDS], dtype=float)
IM_PARAMETER_HIGHS = torch.tensor([b[1] for b in IM_PARAMETER_BOUNDS], dtype=float)

# X~N(0,1)
# Y = normcdf(X)
# transform x -> y as uniform distribution. Do this in the simulation as a separate cumulative function
# Inverse of CDM

IM_PRIOR_DISTRIBUTION = utils.BoxUniform(low=IM_PARAMETER_LOWS, high=IM_PARAMETER_HIGHS)
IM_RANDOM_SAMPLE = IM_PRIOR_DISTRIBUTION.sample()
IM_NUMBER_OF_ROUNDS = 2
IM_NUMBER_OF_SIMULATIONS = 5000
IM_POSTERIOR_MODEL_ESTIMATOR = 'maf'
IM_POSTERIOR_MODEL_HIDDEN_LAYERS = 12
IM_SAVE_X0 = None
IM_GRAPHING_LABELS = [r'y',r'z',r'h',r'$\phi$',r'soma radius',r'trunk length',r'trunk radius',r'basal radius',r'tuft radius',r'basal length']