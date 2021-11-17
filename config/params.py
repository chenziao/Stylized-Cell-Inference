import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

#Dependencies
import numpy as np
import sbi.utils as utils
import torch
import h5py
from torch.distributions.multivariate_normal import MultivariateNormal
from sbi.utils.user_input_checks_utils import MultipleIndependent
import pyro.distributions as dists
import torch

#Project Imports
import config.paths as paths
from stylized_module.feature_extraction.SummaryNet import SummaryNet, SummaryNet3D
from utils.transform.distribution_transformation import range2logn


ACTIVE_CELL = False

#GENERAL PARAMETERS USED ACROSS RUNS
# ELECTRODE_POSITION = np.column_stack((np.zeros(96),np.linspace(-1900,1900,96),np.zeros(96)))
hf = h5py.File(paths.ELECTRODES,'r')
elec_pos = np.array(hf.get('coord'))
ELECTRODE_POSITION = np.column_stack((elec_pos,np.zeros(elec_pos.shape[0])))
ELECTRODE_GRID = (np.array(hf.get('grid/x')),np.array(hf.get('grid/y')),np.zeros(1))



#GET GROUND TRUTH FROM ACTIVE MODEL PARAMS ARE DENOTED WITH THE PREFIX GT
GT_TSTOP = 20. # ms
GT_DT = 0.025  # ms. does not allow change
GT_ELECTRODE_POSITION = ELECTRODE_POSITION #np.column_stack((np.zeros(96),np.linspace(-1900,1900,96),np.zeros(96)))
                        # x,     y,  z,    alpha, h,  phi  # y,   d,  theta,   alpha, h,  phi
GT_LOCATION_PARAMETERS = [0.685,0,49.995,1.43, 1.0, 0.8] #[0, 50, np.pi/4, 1.43, 0.0, 0.8]
GT_GMAX = 0.005
GT_SCALE = 1.
GT_BUTTERWORTH_ORDER = 2 #2nd order
GT_CRITICAL_FREQUENCY = 100 #100 Hz
GT_BANDFILTER_TYPE = 'hp' #highpass
GT_FILTER_SAMPLING_RATE = 40000 #40 kHz


#SIMULATION OF PASSIVE MODEL PARAMS ARE DENOTED WITH THE PREFIX PM
PM_TSTOP = 5. # ms
PM_DT = 0.025 # ms. does not allow change
PM_ELECTRODE_POSITION = ELECTRODE_POSITION #np.column_stack((np.zeros(96),np.linspace(-1900,1900,96),np.zeros(96)))
PM_START_IDX = 320
PM_WINDOW_SIZE = 96
PM_SCALING_FACTOR = 1085. #2.55 


#SIMULATION OF ACTIVE MODEL PARAMS ARE DENOTED WITH THE PREFIX AM
AM_TSTOP = 20.
AM_DT = 0.025
AM_ELECTRODE_POSITION = ELECTRODE_POSITION #np.column_stack((np.zeros(96),np.linspace(-1900,1900,96),np.zeros(96)))
AM_START_IDX = 320
AM_WINDOW_SIZE = 96
AM_TRUE_LOCATION_PARAMETERS = [0,350,80,3.0,0.9,1.27]
AM_FIXED_LOCATION_PARAMETERS = GT_LOCATION_PARAMETERS #[0,380,75,2.9,0.92,1.2]


#INFERENCE MODEL PARAMS ARE DENOTED WITH THE PREFIX IM
IM_BUTTERWORTH_ORDER = 2 #2nd order
IM_CRITICAL_FREQUENCY = 100 #100 Hz
IM_BANDFILTER_TYPE = 'hp' #highpass
IM_FILTER_SAMPLING_RATE = 40000 #40 kHz
IM_Y_DISTANCE = GT_ELECTRODE_POSITION[:,1].ravel()
IM_EMBEDDED_NETWORK = SummaryNet3D(IM_Y_DISTANCE.size, PM_WINDOW_SIZE)
IM_ALPHA_BOUNDS = [0,np.pi]

IM_PARAMETER_BOUNDS = [
    [torch.Tensor([-500]),torch.Tensor([500])],           #y
    [torch.Tensor([20]),torch.Tensor([200])],               #d
    [torch.Tensor([-(np.pi/3)]),torch.Tensor([np.pi/3])],   #theta
    [torch.Tensor([-1]),torch.Tensor([1])],                 #h
    [torch.Tensor([0]),torch.Tensor([np.pi])],              #phi
    [
        torch.Tensor([(np.log(3)+np.log(12))/2]),
        torch.Tensor([(np.log(12)-np.log(3))/6])
    ],                      #r_s
    [torch.Tensor([20]), torch.Tensor([800])],              #l_t
    [
        torch.Tensor([(np.log(0.2)+np.log(1.0))/2]),
        torch.Tensor([(np.log(1.0)-np.log(0.2))/4])
    ],                      #r_t
    [
        torch.Tensor([(np.log(0.2)+np.log(1.0))/2]),
        torch.Tensor([(np.log(1.0)-np.log(0.2))/4])
    ],                      #r_d
    [
        torch.Tensor([(np.log(0.2)+np.log(1.0))/2]),
        torch.Tensor([(np.log(1.0)-np.log(0.2))/4])
    ],                      #r_tu
    [
        torch.Tensor([(np.log(100)+np.log(300))/2]),
        torch.Tensor([(np.log(300)-np.log(100))/4])
    ]                       #l_d
]

# PRIOR_LIST = [
#     # dists.Uniform(IM_PARAMETER_BOUNDS[0][0], IM_PARAMETER_BOUNDS[0][1]),        #y
# #     dists.Uniform(IM_PARAMETER_BOUNDS[1][0], IM_PARAMETER_BOUNDS[1][1]),        #d
# #     dists.Uniform(IM_PARAMETER_BOUNDS[2][0], IM_PARAMETER_BOUNDS[2][1]),        #theta
# #     dists.Uniform(IM_PARAMETER_BOUNDS[3][0], IM_PARAMETER_BOUNDS[3][1]),        #h
# #     dists.Uniform(IM_PARAMETER_BOUNDS[4][0], IM_PARAMETER_BOUNDS[4][1]),        #phi
#     dists.LogNormal(IM_PARAMETER_BOUNDS[5][0], IM_PARAMETER_BOUNDS[5][1]),      #r_s
#     dists.Uniform(IM_PARAMETER_BOUNDS[6][0], IM_PARAMETER_BOUNDS[6][1]),        #l_t
#     # dists.LogNormal(IM_PARAMETER_BOUNDS[7][0], IM_PARAMETER_BOUNDS[7][1]),      #r_t
#     # dists.LogNormal(IM_PARAMETER_BOUNDS[8][0], IM_PARAMETER_BOUNDS[8][1]),      #r_d
#     # dists.LogNormal(IM_PARAMETER_BOUNDS[9][0], IM_PARAMETER_BOUNDS[9][1]),      #r_tu
#     # dists.LogNormal(IM_PARAMETER_BOUNDS[10][0], IM_PARAMETER_BOUNDS[10][1]),    #l_d    
# ]


IM_PRIOR_DISTRIBUTION = dists.Uniform(IM_PARAMETER_BOUNDS[6][0], IM_PARAMETER_BOUNDS[6][1])#MultipleIndependent(PRIOR_LIST, validate_args=False)

# IM_PARAMETER_LOWS = torch.tensor([b[0] for b in IM_PARAMETER_BOUNDS], dtype=float)
# IM_PARAMETER_HIGHS = torch.tensor([b[1] for b in IM_PARAMETER_BOUNDS], dtype=float)
# IM_PRIOR_DISTRIBUTION = utils.BoxUniform(low=IM_PARAMETER_LOWS, high=IM_PARAMETER_HIGHS)
# IM_LOC_PRIOR_DISTRIBUTION = utils.BoxUniform(low=IM_PARAMETER_LOWS, high=IM_PARAMETER_HIGHS)
# IM_GEO_PRIOR_DISTRIBUTION = MultivariateNormal(loc=torch.zeros(6), covariance_matrix=torch.diag(torch.ones(6)))
# IM_PRIOR_DISTRIBUTION = StackedDistribution(IM_LOC_PRIOR_DISTRIBUTION, IM_GEO_PRIOR_DISTRIBUTION)

# IM_RANDOM_SAMPLE = IM_PRIOR_DISTRIBUTION.sample()
IM_NUMBER_OF_ROUNDS = 2
IM_NUMBER_OF_SIMULATIONS = 500
IM_POSTERIOR_MODEL_ESTIMATOR = 'mdn'
IM_POSTERIOR_MODEL_HIDDEN_LAYERS = 12
IM_SAVE_X0 = None
IM_GRAPHING_LABELS = [r'y',r'd',r'theta',r'h',r'$\phi$',r'soma radius',r'trunk length',r'trunk radius',r'basal radius',r'tuft radius',r'basal length']