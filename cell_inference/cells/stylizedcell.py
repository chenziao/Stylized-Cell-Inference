from abc import ABC, abstractmethod
from neuron import h
import math
import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Dict, Union, TypeVar
from enum import Enum

from cell_inference.utils.currents.currentinjection import CurrentInjection

_T = TypeVar("_T", bound=Sequence)

h.load_file('stdrun.hoc')


class CellTypes(Enum):
    PASSIVE = 1
    ACTIVE = 2


class StylizedCell(ABC):
    def __init__(self, geometry: Optional[pd.DataFrame] = None,
                 dl: int = 30, vrest: float = -70.0, nbranch: int = 4) -> None:
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        dL: maximum segment length
        vrest: reversal potential of leak channel for all segments
        nbranch: number of branches of each non-axial section
        """
        self._h = h
        self._dL = dl
        self._vrest = vrest
        self._nbranch = max(nbranch, 2)
        self._nsec = 0
        self._nseg = 0
        self.all = []  # list of all sections
        self.segments = []  # list of all segments
        self.sec_id_lookup = {}  # dictionary from section type id to section index
        self.sec_id_in_seg = []
        self.injection = []
        self.geometry = None
        self.soma = None
        self.set_geometry(geometry)
        self.__setup_all()
        self.seg_coords = self.__calc_seg_coords()

    @abstractmethod
    def set_channels(self) -> str:
        """Abstract method for setting biophysical properties, inserting channels"""
        raise NotImplementedError("Biophysical Channel Properties must be set!")

    @staticmethod
    def set_location(sec, pt0: List[float], pt1: List[float], nseg: int) -> None:
        sec.pt3dclear()
        sec.pt3dadd(*pt0, sec.diam)
        sec.pt3dadd(*pt1, sec.diam)
        sec.nseg = nseg

    def add_injection(self, sec_index, **kwargs):
        """Add current injection to a section by its index"""
        self.injection.append(CurrentInjection(self, sec_index, **kwargs))

    #  PRIVATE METHODS
    def __calc_seg_coords(self) -> Dict:
        """Calculate segment coordinates for ECP calculation"""
        p0 = np.empty((self._nseg, 3))
        p1 = np.empty((self._nseg, 3))
        p05 = np.empty((self._nseg, 3))
        r = np.empty(self._nseg)
        for isec, sec in enumerate(self.all):
            iseg = self.sec_id_in_seg[isec]
            nseg = sec.nseg
            pt0 = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
            pt1 = np.array([sec.x3d(1), sec.y3d(1), sec.z3d(1)])
            pts = np.linspace(pt0, pt1, 2 * nseg + 1)
            p0[iseg:iseg + nseg, :] = pts[:-2:2, :]
            p1[iseg:iseg + nseg, :] = pts[2::2, :]
            p05[iseg:iseg + nseg, :] = pts[1:-1:2, :]
            r[iseg:iseg + nseg] = sec.diam / 2
        return {'dl': p1 - p0, 'pc': p05, 'r': r}

    def __create_morphology(self) -> None:
        """Create cell morphology"""
        if self.geometry is None:
            raise ValueError("Warning: geometry is not loaded.")
        self._nsec = 0
        rot = 2 * math.pi / self._nbranch
        for sec_id, sec in self.geometry.iterrows():
            start_idx = self._nsec
            if sec_id == 0:
                r0 = sec['R']
                pt0 = [0., -2 * r0, 0.]
                pt1 = [0., 0., 0.]
                self.soma = self.create_section(name=sec['name'], diam=2 * r0)
                self.set_location(self.soma, pt0, pt1, 1)
            else:
                length = sec['L']
                radius = sec['R']
                ang = sec['ang']
                nseg = math.ceil(length / self._dL)
                pid = self.sec_id_lookup[sec['pid']][0]
                psec = self.all[pid]
                pt0 = [psec.x3d(1), psec.y3d(1), psec.z3d(1)]
                if sec['axial']:
                    nbranch = 1
                    x = 0
                    pt1[1] = pt0[1] + length
                else:
                    nbranch = self._nbranch
                    x = length * math.cos(ang)
                    pt1[1] = pt0[1] + length * math.sin(ang)
                for i in range(nbranch):
                    pt1[0] = pt0[0] + x * math.cos(i * rot)
                    pt1[2] = pt0[2] + x * math.sin(i * rot)
                    section = self.create_section(name=sec['name'], diam=2 * radius)
                    section.connect(psec(1), 0)
                    self.set_location(section, pt0, pt1, nseg)
            self.sec_id_lookup[sec_id] = list(range(start_idx, self._nsec))
        self.set_location(self.soma, [0., -r0, 0.], [0., r0, 0.], 1)
        self.__store_segments()

    def __setup_all(self) -> None:
        if self.geometry is not None:
            self.__create_morphology()
            self.set_channels()

    def __store_segments(self) -> None:
        self.segments = []
        self.sec_id_in_seg = []
        nseg = 0
        for sec in self.all:
            self.sec_id_in_seg.append(nseg)
            nseg += sec.nseg
            for seg in sec:
                self.segments.append(seg)
        self._nseg = nseg

    #  PUBLIC METHODS
    def create_section(self, name: str = 'null_sec', diam: float = 500.0) -> h.Section:
        sec = h.Section(name=name, cell=self)
        sec.diam = diam
        self.all.append(sec)
        self._nsec += 1
        return sec

    def get_sec_by_id(self, index: Optional[_T] = None) -> Optional[Union[List[h.Section], h.Section]]:
        """Get section(s) objects by index(indices) in the section list"""
        if not hasattr(index, '__len__'):
            sec = self.all[index]
        else:
            sec = [self.all[i] for i in index]
        return sec

    def get_seg_by_id(self, index: Optional[_T] = None) -> List[h.Section]:
        """Get segment(s) objects by index(indices) in the segment list"""
        if not hasattr(index, '__len__'):
            seg = self.segments[index]
        else:
            seg = [self.segments[i] for i in index]
        return seg

    def set_all_passive(self, gl: float = 0.0003) -> None:
        """A use case of 'set_channels', set all sections passive membrane"""
        for sec in self.all:
            sec.cm = 1.0
            sec.insert('pas')
            sec.g_pas = gl
            sec.e_pas = self._vrest

    def set_geometry(self, geometry: Optional[pd.DataFrame] = None) -> None:
        if geometry is None:
            self.geometry = None
        else:
            if not isinstance(geometry, pd.DataFrame):
                raise TypeError("geometry must be a pandas dataframe")
            if geometry.iloc[0]['type'] != 1:
                raise ValueError("first row of geometry must be soma")
            self.geometry = geometry.copy()
