import numpy as np
from neuron import hoc
from matplotlib.axes import Axes

from typing import Optional

def measure_passive_properties(V, t, iclamp: hoc.HocObject, ax: Optional[Axes] = None):
    """
    V: membrane voltage response to current injection (mV)
    t: time points (ms)
    iclamp: IClamp object that specifies the current injection (nA)
    ax: axes for plotting V vs. t
    Return passive properties: resting potential (mV), input resistance (megaohms), time constant (ms)
    """
    t = np.asarray(t).ravel()
    V = np.asarray(V).ravel()
    amp = iclamp.amp
    delay = iclamp.delay
    dur = iclamp.dur
    idx = np.searchsorted(t,[delay,delay+dur])
    idx[0] = max(idx[0]-1,0)
    V = V[idx[0]:idx[1]]
    t = t[idx[0]:idx[1]]
    if ax is not None:
        ax.plot(t,V)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('membrane voltage (mV)')
    Vrest = V[0] # mV
    deltaV = V[-1]-V[0]
    Rin = deltaV/amp # megohms
    dv = np.abs(V-Vrest)
    V_tau = np.abs(V[-1]-deltaV/np.e-Vrest)
    idx = np.nonzero((dv[:-1]<=dv[1:])|(dv[:-1]<=V_tau))[0]
    dv = dv[np.insert(idx+1,0,0)]
    tau_idx = np.searchsorted(dv,V_tau)
    if tau_idx==0 or tau_idx==dv.size:
        raise ValueError("Passive properties measurement failed. Voltage response is not monotone during current injection.")
    Tau = t[tau_idx]-t[0] # ms
    return Vrest, Rin, Tau