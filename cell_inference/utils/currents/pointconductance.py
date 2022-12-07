from __future__ import annotations

from neuron import h
from typing import TYPE_CHECKING

from cell_inference.utils.currents.pointcurrent import DensePointCurrent

if TYPE_CHECKING:
    from cell_inference.cells.stylizedcell import StylizedCell

class PointConductance(DensePointCurrent):
    def __init__(self, cell: StylizedCell, sec_index: int, L_unit: float = 1., 
                 dens_params: dict = {'g_e0': 1e-5, 'g_i0': 3e-5, 'std_e': 1., 'std_i': 2.},
                 cnst_params: dict = {'tau_e': 3., 'tau_i': 15.}, record: bool = False):
        super().__init__(cell, sec_index)
        self.dens_params = dens_params # conductance/unit length, std relative to conductance
        self.L_unit = L_unit # unit length (um)
        self.cnst_params = cnst_params # constant parameters
        self.setup(record)
    
    def __setup_Gfluct(self):
        for seg in self.get_section():
            self.pp_obj.append(h.Gfluct2(seg))
        self.set_params()

    # PUBLIC METHODS
    def setup(self, record: bool = False):
        self.__setup_Gfluct()
        if record:
            self.setup_recorder()

    def set_params(self, **kwargs):
        """kwargs: initialization keyword arguments """
        for key, value in kwargs.items():
            setattr(self, key, value) 
        dens = self.segment_length() / self.L_unit
        params = self.dens_params.copy()
        params['g_e0'] *= dens
        params['g_i0'] *= dens
        params['std_e'] *= params['g_e0']
        params['std_i'] *= params['g_i0']
        params.update(self.cnst_params)
        for g in self.pp_obj:
            for key, value in params.items():
                setattr(g, key, value)