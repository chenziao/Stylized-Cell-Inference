import numpy as np
import torch
from scipy.interpolate import griddata
from typing import Tuple

#Project Imports
import config.params as params
from utils.spike_window import first_pk_tr


def Grid_LFP(lfp: np.ndarray,
             coord: np.ndarray,
             grid_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a grid to match the Neuropixel Probe used to collect
    LFP signals: https://www.neuropixels.org/probe
    
    Then interpolate the LFP onto the grid for each timestep
    """
    t = lfp.shape[0]
    xy = coord[:,:2]
    xx, yy = np.meshgrid(grid_v[0],grid_v[1],indexing='ij')
    grid = np.column_stack((xx.ravel(),yy.ravel()))
    grid_lfp = np.empty((t,grid.shape[0]))
    for i in range(t):
        grid_lfp[i,:] = griddata(xy,lfp[i,:],grid)
    return grid_lfp, grid


def Stats(g_lfp: np.ndarray,
          grid: np.ndarray=None) -> np.ndarray:
    """
    Calculates summary statistics. This includes:
        - Average Voltage of Each Channel
        - Standard Deviation Voltage of Each Channel
        - Troughs of Each Channel
        - Peaks of Each Channel
        - Difference between Peak and Trough amplitudes
        - Width of Waveform from half height (if grid specified)
    """
    g_lfp = np.asarray(g_lfp)
    grid_shape = tuple(v.size for v in params.ELECTRODE_GRID[:2])
    avg = np.mean(g_lfp,axis=0) # average voltage of each channel
    stdDev = np.std(g_lfp,axis=0) # stDev of the voltage of each channel
    tT = np.argmin(g_lfp,axis=0)
    tP = np.argmax(g_lfp,axis=0)
    
    Troughs = -np.take_along_axis(g_lfp,np.expand_dims(tT,axis=0),axis=0)
    Peaks = np.take_along_axis(g_lfp,np.expand_dims(tP,axis=0),axis=0)
    relT = tP-tT
    
    stats_list = [avg,relT,stdDev,Troughs,Peaks]
    I_min = 2 # include minimum statistics for the the first I_min in stats_list
    
    
    """
    Helper functions for calculating statistics across channels and searching for
    a specified height
        - statscalc: calculates the statistics across each channel
            - include_min: include minimun value and position
            
        - searchheights: calculates width of waveform given a height
    """
    def statscalc(stats,include_min=True):
        stats = stats.ravel()
        mean = np.mean(stats)
        std = np.std(stats)
        M = np.argmax(stats)
        max_val = stats[M]
        Mx, My = np.unravel_index(M,grid_shape)
        if include_min:
            m = np.argmin(stats)
            min_val = stats[m]
            mx, my = np.unravel_index(m,grid_shape)
            All = np.array([mean,std,Mx,My,max_val,mx,my,min_val])
        else:
            All = np.array([mean,std,Mx,max_val])#My,max_val])
        return All
    
    def searchheights(lfp, height, idx):
        idx_left, idx_right = 0, lfp.size
        for i in range(idx, idx_left, -1):
            if lfp[i] <= height:
                idx_left = i
                break
        for i in range(idx, idx_right):
            if lfp[i] <= height:
                idx_right = i
                break
        return idx_left, idx_right
    
    #Calculation of statistics across channels
#     sl = [statscalc(x,i<I_min) for i,x in enumerate(stats_list)]
    sl = []
    
    """
    Calculates width of the first peak and adds it to the stats
    if grid parameter is provided
    """
    if grid is not None:
        t0 = first_pk_tr(g_lfp)
        reshaped_lfp = (g_lfp[t0,:].reshape(4,190))
        x0 = np.argmax(np.max(np.abs(reshaped_lfp), axis=1), axis=0)
        fy = reshaped_lfp[x0,:]
        y0 = np.argmax(np.abs(fy), axis=0)
        half_height = np.abs(fy[y0])/2
        min_idx, max_idx = searchheights(np.abs(fy), half_height, y0)
        sl += [np.array([max_idx,min_idx])]

        
    allStats = np.concatenate(sl)
    return allStats



def cat_output(lfp: np.ndarray, 
               include_sumstats=True) -> torch.Tensor:
    """
    Driver function. Creates grid and interpolates provided LFP then concatenates
    calculated summary statistics to the interpolated LFP

    Always include summary stats unless specified not to
    """
    g_lfp, grid = Grid_LFP(lfp,params.ELECTRODE_POSITION,params.ELECTRODE_GRID)
    output = np.concatenate((g_lfp.ravel(),Stats(g_lfp, grid))) if include_sumstats else lfp.ravel()
    return torch.from_numpy(output)
