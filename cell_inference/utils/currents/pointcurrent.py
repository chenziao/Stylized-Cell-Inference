from neuron import h
from abc import ABC, abstractmethod

from cell_inference.utils.currents.recorder import Recorder


class PointCurrent(ABC):
    """A module for current point process"""

    def __init__(self, cell, sec_index, loc=0.5):
        """
        cell: target cell object
        sec_index: index of the target section in the section list
        loc: location on a section, between [0,1]
        """
        self.cell = cell
        self.sec_index = sec_index
        self.loc = loc
        self.pp_obj = None  # point process object
        self.rec_vec = None  # vector for recording

    @abstractmethod
    def setup(self, record: bool = None) -> None:
        pass

    def setup_recorder(self):
        size = [round(h.tstop / h.dt) + 1] if hasattr(h, 'tstop') else []
        self.rec_vec = h.Vector(*size).record(self.pp_obj._ref_i)

    def get_section(self) -> h.Section:
        return self.cell.all[self.sec_index]

    def get_segment(self):
        return self.pp_obj.get_segment()

    def get_segment_id(self) -> int:
        """Get the index of the injection target segment in the segment list"""
        iseg = int(self.get_segment().x * self.get_section().nseg)
        return self.cell.sec_id_in_seg[self.sec_index] + iseg


class DensePointCurrent(ABC):
    """A module for current point process inserted per segment"""

    def __init__(self, cell, sec_index):
        """
        cell: target cell object
        sec_index: index of the target section in the section list
        """
        self.cell = cell
        self.sec_index = sec_index
        self.pp_obj = []  # list of point process objects
        self.rec_vec = None  # recorder object for current recording

    @abstractmethod
    def setup(self, record: bool = None) -> None:
        pass

    def setup_recorder(self):
        size = [round(h.tstop / h.dt) + 1] if hasattr(h, 'tstop') else []
        self.rec_vec = Recorder(self.pp_obj, 'i')

    def get_section(self) -> h.Section:
        return self.cell.all[self.sec_index]

    def get_segment(self, seg_idx = 0):
        return self.pp_obj[seg_idx].get_segment()

    def segment_length(self) -> float:
        """Segment length (um)"""
        sec = self.get_section()
        return sec.L / sec.nseg