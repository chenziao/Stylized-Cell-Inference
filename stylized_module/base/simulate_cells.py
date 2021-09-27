import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

from torch import Tensor
import numpy as np
from numpy import ndarray

import config.params as params
import config.paths as paths
from stylized_module.base.active_model_synapse_input import Simulation as amSimulation
from stylized_module.base.passive_model_soma_injection import Simulation as pmSimulation
from stylized_module.models.SummaryStats2D import cat_output
from utils.transform.distribution_transformation import norm2unif, range2logn, norm2logn, logds_norm2unif, logds_norm2logn

rng = np.random.default_rng(123412)


def run_am_simulation(data_path: str=paths.SIMULATED_DATA_FILE) -> tuple(amSimluation, int, ndarray, ndarray):
    """
    The function takes in the specific data file and returns simulation results in the form of a tuple.

    Args:
        data_path: String with a path pointing to the data location
    
    Returns:
        Tuple of 4 items:
            sim: The simulation instance
            window_size: The size of the window used in the simulation
            x0_trace: The initial LFP trace
            t0: The LFP trace time

    """

    #NEURON Simulation Parameters and Loading
    h.nrn_load_dll(paths.COMPILED_LIBRARY)
    h.tstop = params.AM_TSTOP
    h.dt = params.AM_DT

    #Loading Cell Parameters
    geo_standard = pd.read_csv(paths.GEO_STANDARD,index_col='id')
    hf = h5py.File(data_path, 'r')
    groundtruth_lfp = np.array(hf.get('data'))
    hf.close()
    x0_trace = groundtruth_lfp[params.AM_START_IDX:params.AM_START_IDX+params.AM_WINDOW_SIZE,:]

    #Setting Simulation Runtime Parameters and Running
    sim = amSimulation(geo_standard,params.AM_ELECTRODE_POSITION,loc_param=params.AM_FIXED_LOCATION_PARAMETERS)
    sim.run()
    t = sim.t()
    t0 = t[:params.AM_WINDOW_SIZE]
    window_size = params.AM_WINDOW_SIZE

    return sim, window_size, x0_trace, t0


def run_pm_simulation(data_path: str=paths.SIMULATED_DATA_FILE) -> tuple(pmSimluation, int, ndarray, ndarray):
    """
    The function takes in the specific data file and returns simulation results in the form of a tuple.

    Args:
        data_path: String with a path pointing to the data location
    
    Returns:
        Tuple of 4 items:
            sim: The simulation instance
            window_size: The size of the window used in the simulation
            x0_trace: The initial LFP trace
            t0: The LFP trace time

    """

    #NEURON Simulation Parameters and Loading
    h.nrn_load_dll(paths.COMPILED_LIBRARY)
    h.tstop = params.PM_TSTOP
    h.dt = params.PM_DT

    #Loading Cell Parameters
    geo_standard = pd.read_csv(paths.GEO_STANDARD,index_col='id')
    hf = h5py.File(data_path, 'r')
    groundtruth_lfp = np.array(hf.get('data'))
    hf.close()

    #Setting Simulation Runtime Parameters and Running
    window_size = params.PM_WINDOW_SIZE
    maxIndx = np.argmax(np.absolute(groundtruth_lfp).max(axis=0))  # find maximum absolute value from averaged traces
    maxTrace = -groundtruth_lfp[params.PM_START_IDX:,maxIndx]
    x0_trace = groundtruth_lfp[params.PM_START_IDX:params.PM_START_IDX+params.PM_WINDOW_SIZE,:]
    soma_injection = np.insert(maxTrace,0,0.)
    soma_injection = np.asarray([s * params.PM_SCALING_FACTOR for s in soma_injection])
    sim = pmSimulation(geo_standard,params.PM_ELECTRODE_POSITION,soma_injection)
    sim.run()
    t = sim.t()
    t0 = t[:params.PM_WINDOW_SIZE]

    return sim, window_size, x0_trace, t0


def run_sim_from_sample(param: Tensor,
                        sim: amSimulation,
                        cell_type: str='active', 
                        input_coordinates: str='polar',
                        whole_trace: bool=False,
                        **kwargs
) -> ndarray:
"""
The function takes in a sample and converts it to the correct parameter space. Once done it
simulates a cell simulation with the specified parameters and returns the resulting filtered LFP

Args:
    param: Tensor sample of data provided corresponding to cell location and geometric parameters
    sim: Simulation instance from active cell #TODO add passive cell simulation instances
    cell_type: Version of cell, currently only working with active. #TODO implement 'passive'
    input_coordinates: Type of sample parameters to expect as input #TODO implement euclidean
    whole_trace: specify whether to return the entire LFP trace or not
    kwargs: additional arguments that are not currently relevant for the users

Returns:
    Filtered LFP in a Numpy Array

"""

    #Build Butterworth Filter
    filt_b,filt_a = signal.butter(params.IM_BUTTERWORTH_ORDER,
                              params.IM_CRITICAL_FREQUENCY,
                              params.IM_BANDFILTER_TYPE,
                              fs=params.IM_FILTER_SAMPLING_RATE)

    #Replace alpha with random uniform number
    alpha = rng.uniform(low=params.IM_ALPHA_BOUNDS[0], high=params.IM_ALPHA_BOUNDS[1])

    #convert polar to euclidean
    d = norm2unif(param[1], params.IM_PARAMETER_BOUNDS[1][0], params.IM_PARAMETER_BOUNDS[1][1])
    theta = norm2unif(param[2], params.IM_PARAMETER_BOUNDS[2][0], params.IM_PARAMETER_BOUNDS[2][1])
    x = d * np.sin(theta)
    z = d * np.cos(theta)
    
    #organizing and setting simulation location parameters
    numpy_list = np.array([
        x,                                                                                       #x
        norm2unif(param[0], params.IM_PARAMETER_BOUNDS[0][0], params.IM_PARAMETER_BOUNDS[0][1]), #y
        z,                                                                                       #z
        alpha,                                                                                   #alpha
        norm2unif(param[3], params.IM_PARAMETER_BOUNDS[3][0], params.IM_PARAMETER_BOUNDS[3][1]), #h
        norm2unif(param[4], params.IM_PARAMETER_BOUNDS[4][0], params.IM_PARAMETER_BOUNDS[4][1])  #phi
    ])
    sim.set_loc_param(torch.from_numpy(numpy_list))
    
    #organizing and setting simulation geometric parameters
    geo_list = np.zeros(6)
    geo_list[0] = norm2unif(param[5], params.IM_PARAMETER_BOUNDS[5][0], params.IM_PARAMETER_BOUNDS[5][1])
    for i in range(6,11):
        if i == 6:
            m,s=range2logn(params.IM_PARAMETER_BOUNDS[i][0], params.IM_PARAMETER_BOUNDS[i][1], n_sigma=3)
        else:
            m,s=range2logn(params.IM_PARAMETER_BOUNDS[i][0], params.IM_PARAMETER_BOUNDS[i][1])
        geo_list[i-5] = norm2logn(param[i], m, s)
    sim.set_geo_param(torch.from_numpy(geo_list))

    #simluation specific run time parameters
    scalVal = 1
    sim.set_scale(scalVal)
    sim.set_gmax(params.GT_GMAX)
    sim.set_scale(scalVal)
    sim.create_cells()
    sim.run()

    #extract lfp, filter, and return
    lfp = sim.get_lfp().T
    filtered_lfp = signal.lfilter(filt_b,filt_a,lfp,axis=0) # filter along row of the lfp 2d-array, if each row is a channel
    if not whole_trace:
        start,end = get_spike_window(filtered_lfp,win_size=params.AM_WINDOW_SIZE,align_at=fst_idx)
        filtered_lfp = filtered_lfp[start:end,:]
    return filtered_lfp


def simulate(sim_params: Tensor, sim: Optional[amSimulation]) -> Tensor:
    """
    The function drives the entire simulation, it takes the simulated parameters and concatenates them
    calculated Summary Statistics

    Args:
        sim_params: The sampled simulation cell parameters that get passed directly to the simulation
        sim: The simulation instance #TODO ensure this works with passive cells

    Returns:
        The Tensor of the concatenated simulated LFPs and Summary Statistics
        
    """
    if params.ACTIVE_CELL is False:
        lfp = run_sim_from_sample(sim_params, sim, cell_type='passive')
    else:
        lfp = run_sim_from_sample(sim_params, sim, cell_type='active')
    return cat_output(lfp)