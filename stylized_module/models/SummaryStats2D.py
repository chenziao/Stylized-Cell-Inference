import numpy as np
#Project Imports
import config.params as params

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
    grid_shape = tuple(v.size for v in params.ELECTRODE_GRID[:2])
    
    avg = np.mean(lfp,axis=0) # average voltage of each channel
    stdDev = np.std(lfp,axis=0) # stDev of the voltage of each channel
    tT = np.argmin(lfp,axis=0)
    tP = np.argmax(lfp,axis=0)
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
            All = np.array([mean,std,Mx,My,max_val])
        return All
    
    allStats = np.concatenate([statscalc(x,i<I_min) for i,x in enumerate(stats_list)])
    return allStats

def cat_output(lfp):
    lfp,_ = Grid_LFP(lfp,params.ELECTRODE_POSITION,params.ELECTRODE_GRID)
    output = np.concatenate((lfp.ravel(),Stats(lfp)))
    return torch.from_numpy(output)
