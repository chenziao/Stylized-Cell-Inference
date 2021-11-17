import numpy as np
import torch
from scipy.interpolate import griddata

#Project Imports
import config.params as params
from utils.spike_window import first_pk_tr

def Grid_LFP(lfp,coord,grid_v):
    t = lfp.shape[0]
    xy = coord[:,:2]
    xx, yy = np.meshgrid(grid_v[0],grid_v[1],indexing='ij')
    grid = np.column_stack((xx.ravel(),yy.ravel()))
    grid_lfp = np.empty((t,grid.shape[0]))
    for i in range(t):
        grid_lfp[i,:] = griddata(xy,lfp[i,:],grid)
    return grid_lfp, grid

def Stats(lfp):
    """
    Calculates summary statistics
    """
    lfp = np.asarray(lfp)
#     print(lfp.shape)
    grid_shape = tuple(v.size for v in params.ELECTRODE_GRID[:2])
    avg = np.mean(lfp,axis=0) # average voltage of each channel
    stdDev = np.std(lfp,axis=0) # stDev of the voltage of each channel
    tT = np.argmin(lfp,axis=0)
    tP = np.argmax(lfp,axis=0)
    
    t0 = first_pk_tr(lfp)
    reshaped_lfp = (lfp[t0,:].reshape(4,190))
    x0 = np.argmax(np.max(np.abs(reshaped_lfp), axis=1), axis=0)
#     print(x0)
    fy = reshaped_lfp[x0,:]
#     fy = np.take(fy,x0)
#     print(fy.shape)
    y0 = np.argmax(np.abs(fy), axis=0)
#     print(fy)
    half_height = np.abs(fy[y0])/2
    
    def searchheights(lfp, height, idx):
        i, j = idx, idx
        while(lfp[i] > height and lfp[j] > height):
            i += 1
            j -= 1
            if i >= 190 or j <= 0:
                return 0, 190
        while(lfp[i] > height):
            i += 1
            if i >= 190 or j <= 0:
                return 0, 190
        while(lfp[j] > height):
            j -= 1
            if i >= 190 or j <= 0:
                return 0, 190
        return j, i
    
    min_idx, max_idx = searchheights(np.abs(fy), half_height, y0)
#     print(half_height, min_idx, max_idx, y0)
    
    Troughs = -np.take_along_axis(lfp,np.expand_dims(tT,axis=0),axis=0)
    Peaks = np.take_along_axis(lfp,np.expand_dims(tP,axis=0),axis=0)
    relT = tP-tT
    
    stats_list = [avg,relT,stdDev,Troughs,Peaks]
    I_min = 2 # include minimum statistics for the the first I_min in stats_list
    
    # Statistics across channels
    def statscalc(stats,include_min=True):
        """include_min: include minimun value and position"""
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
    sl = [statscalc(x,i<I_min) for i,x in enumerate(stats_list)]+[np.array([max_idx,min_idx])]
#     print(sl)
    allStats = np.concatenate(sl)
    return allStats

def cat_output(lfp):
    lfp,_ = Grid_LFP(lfp,params.ELECTRODE_POSITION,params.ELECTRODE_GRID)
    output = np.concatenate((lfp.ravel(),Stats(lfp)))
#     output = lfp.ravel()
#     print(torch.from_numpy(output).shape)
    return torch.from_numpy(output)
