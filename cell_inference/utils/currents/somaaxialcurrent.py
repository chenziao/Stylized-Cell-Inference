from neuron import h, nrn
import numpy as np
from cell_inference.utils.currents.recorder import Recorder

from typing import Optional

class Soma_Axial_Current(object):
    def __init__(self, soma: nrn.Section, dend_type: Optional[str] = None, record_t: bool = False, single_seg: bool = False) -> None:
        self.soma = soma
        if dend_type is None:
            sec_names = [sec.name().split('.')[-1].split('[')[0] for sec in soma.children()]
            self.dend_type = list(set(sec_names))
        else:
            self.dend_type = dend_type
        self.dend = {};
        for d in self.dend_type:
            self.dend[d] = Adjacent_Section(self.soma,d)
        self.single_seg = single_seg
        self.setup_recorder(record_t)
    
    def setup_recorder(self, record_t: bool = False):
        if record_t:
            self.t_vec = h.Vector(round(h.tstop / h.dt) + 1).record(h._ref_t)
        else:
            self.t_vec = None
        for dend in self.dend.values():
            dend.setup_recorder(self.single_seg)
    
    def t(self):
        if self.t_vec is None:
            t = None
        else:
            t = self.t_vec.as_numpy().copy()
        return t
    
    def get_current(self, dend_type: Optional[str] = None) -> np.ndarray:
        if dend_type is None:
            axial_current = {}
            for name,dend in self.dend.items():
                axial_current[name] = dend.get_current()
        else:
            axial_current = self.dend[dend_type].get_current()
        return axial_current

class Adjacent_Section(object):
    def __init__(self, soma: nrn.Section, name: Optional[str] = 'dend') -> None:
        self.name = name
        self.init_sec = [s for s in soma.children() if name in s.name()]
        self.nseg = [s.nseg for s in self.init_sec]
        self.init_seg = [s(0.5/n) for s,n in zip(self.init_sec,self.nseg)]
    
    def setup_recorder(self, single_seg: bool = False):
        self.soma_seg = [s.parentseg() for s in self.init_sec]
        if len(set(self.soma_seg)) == 1 and len(self.soma_seg)>1:
            self.soma_seg = [self.soma_seg[0]]
        if single_seg:
            self.init_seg = [self.init_seg[0]]
        self.soma_v = Recorder(self.soma_seg)
        self.dend_v = Recorder(self.init_seg)
    
    def get_current(self) -> np.ndarray:
        v_soma = self.soma_v.as_numpy()
        v_dend = self.dend_v.as_numpy()
        axial_r = np.array([[seg.ri()] for seg in self.init_seg])
        axial_current = (v_dend-v_soma)/axial_r
        return axial_current
