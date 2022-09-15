# Dependencies
from neuron import h
import pandas as pd
import numpy as np
from typing import Optional, Union
import warnings

# Project Imports
from cell_inference.cells.stylizedcell import StylizedCell

h.load_file('stdrun.hoc')


class ActiveCell(StylizedCell):
    """Define single cell model using parent class Stylized_Cell"""

    def __init__(self, geometry: Optional[pd.DataFrame] = None,
                 biophys: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        biophys: vector of biophysical parameters corresponding to "biophys_entries". Use -1 for default value.
        dL: maximum segment length
        vrest: reversal potential for leak channels
        """
        self.grp_ids = []
        self.biophys = biophys
        self.grp_sec_type_ids = [[0], [1, 2], [3, 4]]  # select section id's for each group
        self.biophys_entries = [
            (0, 'g_pas'), (1, 'g_pas'), (2, 'g_pas'),  # g_pas of soma, basal, apical
            (0, 'gbar_NaV'), (1, 'gbar_NaV'), (2, 'gbar_NaV'),
            (0, 'gbar_Kv3_1'), (1, 'gbar_Kv3_1'), (2, 'gbar_Kv3_1')
            # (0, 'gNaTa_tbar_NaTa_t'), (2, 'gNaTa_tbar_NaTa_t'),  # gNaTa_t of soma, apical
            # (0, 'gSKv3_1bar_SKv3_1'), (2, 'gSKv3_1bar_SKv3_1')  # gSKv3_1 of soma, apical
        ]
        self.default_biophys = np.array([0.00051532, 0.000170972, 0.004506, 0.0433967, 0.016563, 0.0109506, 0.00639898, 0.0564755, 0.913327])
        # self.default_biophys = np.array([3.3e-5, 6.3e-5, 8.8e-5, 2.43, 0.0252, 0.983, 0.0112])
        # self.default_biophys = np.array([0.0000338, 0.0000467, 0.0000589, 2.04, 0.0213, 0.693, 0.000261])
        
        super().__init__(geometry, **kwargs)
#         self.set_channels()

    # PRIVATE METHODS
    def __create_biophys_entries(self) -> np.ndarray:
        """
        Define list of entries of biophysical parameters.
        Each entry is a pair of group id and parameter reference string.
        Define default values and set parameters in "biophys".
        """
        self.grp_ids = [[isec for i in ids for isec in self.sec_id_lookup[i]] for ids in self.grp_sec_type_ids]
        biophys = self.default_biophys
        if self.biophys is not None:
            for i in range(len(self.biophys)):
                if not np.isnan(self.biophys[i]):
                    biophys[i] = self.biophys[i]
        self.biophys = biophys

    # PUBLIC METHODS
    def set_channels(self) -> None:
        if not self.grp_ids:
            self.__create_biophys_entries()
        # common parameters
        for sec in self.all:
            sec.cm = 2.0
            sec.Ra = 100
            sec.insert('pas')
            sec.e_pas = self._vrest
        # fixed parameters
        soma = self.soma
        soma.cm = 1.0
        soma.insert('NaV')  # Sodium channel
        soma.insert('Kv3_1')  # Potassium channel
        soma.ena = 50
        soma.ek = -85
        for isec in self.grp_ids[2]:
            sec = self.get_sec_by_id(isec)  # apical dendrites
            sec.insert('NaV')
            sec.insert('Kv3_1')
            sec.ena = 50
            sec.ek = -85

        for isec in self.grp_ids[1]:
            sec = self.get_sec_by_id(isec)  # basal dendrites
            sec.insert('NaV')
            sec.insert('Kv3_1')
            sec.ena = 50
            sec.ek = -85
        # variable parameters
        for i, entry in enumerate(self.biophys_entries):
            for sec in self.get_sec_by_id(self.grp_ids[entry[0]]):
                try:
                    setattr(sec, entry[1], self.biophys[i])
                except AttributeError:
                    warnings.warn("Error: {} not found in {}".format(entry[1], sec))

        h.v_init = self._vrest
