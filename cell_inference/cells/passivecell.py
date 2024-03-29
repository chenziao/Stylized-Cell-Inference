# Dependencies
from neuron import h
import pandas as pd
import numpy as np

# Project Imports
from cell_inference.cells.stylizedcell import StylizedCell

h.load_file('stdrun.hoc')


class PassiveCell(StylizedCell):
    """Define single cell model using parent class Stylized_Cell"""

    def __init__(self, geometry: pd.DataFrame = None, **kwargs) -> None:
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        dL: maximum segment length
        vrest: reversal potential of leak channel for all segments
        """
        super().__init__(geometry, **kwargs)
        self.set_channels()

    def set_channels(self) -> None:
        """Define biophysical properties, insert channels"""
        #         self.set_all_passive(gl=0.0003)  # soma,dend both have gl
        gl_soma = 3.3e-5
        gl_dend = 6.3e-5
        for sec in self.all:
            sec.cm = 2.0
            sec.Ra = 100
            sec.insert('pas')
            sec.e_pas = self._vrest
        soma = self.soma
        soma.cm = 1.0
        soma.g_pas = gl_soma
        for sec in self.all[1:]:
            sec.g_pas = gl_dend
        h.v_init = self._vrest
