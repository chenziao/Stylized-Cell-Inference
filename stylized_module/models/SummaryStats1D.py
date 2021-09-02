import numpy as np

def Stats(lfp):
    """
    Calculates summary statistics
    """
    lfp = np.asarray(lfp)
    
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
        if include_min:
            m = np.argmin(stats)
            min_val = stats[m]
            All = np.array([mean,std,M,max_val,m,min_val])
        else:
            All = np.array([mean,std,M,max_val])
        return All
    
    allStats = np.concatenate([statscalc(x,i<I_min) for i,x in enumerate(stats_list)])
    return allStats

def cat_output(lfp):
    output = np.concatenate((lfp.ravel(),Stats(lfp)))
    return torch.from_numpy(output)
