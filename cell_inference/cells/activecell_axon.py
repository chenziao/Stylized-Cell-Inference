# Dependencies
from neuron import h
import pandas as pd
import numpy as np
from typing import Optional, Union
import warnings

# Project Imports
from cell_inference.cells.stylizedcell import StylizedCell
from cell_inference.utils.currents.recorder import Recorder

h.load_file('stdrun.hoc')


class ActiveCellAxon(StylizedCell):
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
        self.v_rec = None
        self.grp_sec_type_ids = [[0], [1, 2], [3, 4], [5]]  # select section id's for each group
        self.biophys_entries = [
            (0, 'g_pas'), (1, 'g_pas'), (2, 'g_pas'),  # g_pas of soma, basal, apical
            (0, 'gbar_NaV'), (1, 'gbar_NaV'), (2, 'gbar_NaV'),
            (0, 'gbar_Kv3_1'), (1, 'gbar_Kv3_1'), (2, 'gbar_Kv3_1'),
            (3, 'g_pas'), 3, 'gbar_NaV'), (3, 'gbar_Kv3_1')
        ]
        
        super(ActiveCell, self).__init__(geometry, **kwargs)
        self.v_rec = self.__record_soma_v()
        
#         self.set_channels()

    # PRIVATE METHODS
    def __create_biophys_entries(self, biophys: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Define list of entries of biophysical parameters.
        Each entry is a pair of group id and parameter reference string.
        Define default values and set parameters in "biophys".
        """
        for ids in self.grp_sec_type_ids:
            secs = []
            for i in ids:
                secs.extend(self.sec_id_lookup[i])
            self.grp_ids.append(secs)
        default_biophys = np.array([0.00051532, 0.000170972, 0.004506, 0.0433967, 0.016563, 0.0109506, 0.00639898, 0.0564755, 0.913327, 0.00951182, 0.000326646, 0.770355])
        #default_biophys = np.array([3.3e-5, 6.3e-5, 8.8e-5, 2.43, 0.0252, 0.983, 0.0112])
        #default_biophys = np.array([0.0000338, 0.0000467, 0.0000589, 2.04, 0.0213, 0.693, 0.000261])
        if biophys is not None:
            for i in range(len(biophys)):
                if biophys[i] >= 0:
                    default_biophys[i] = biophys[i]
        return default_biophys

    def __record_soma_v(self) -> Recorder:
        return Recorder(self.soma(.5), 'v')

    # PUBLIC METHODS
    def set_channels(self) -> None:
        if not self.grp_ids:
            self.biophys = self.__create_biophys_entries(self.biophys)
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
        for isec in self.grp_ids[1]:
            sec = self.get_sec_by_id(isec)  # basal dendrites
            sec.insert('NaV')
            sec.insert('Kv3_1')
            sec.ena = 50
            sec.ek = -85
        for isec in self.grp_ids[2]:
            sec = self.get_sec_by_id(isec)  # apical dendrites
            sec.insert('NaV')
            sec.insert('Kv3_1')
            sec.ena = 50
            sec.ek = -85
        for isec in self.grp_ids[3]:
            sec = self.get_sec_by_id(isec)  # axon
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

    def v(self) -> Optional[Union[str, np.ndarray]]:
        """Return recorded soma membrane voltage in numpy array"""
        if hasattr(self, 'v_rec'):
            return self.v_rec.as_numpy()
        else:
            raise NotImplemented("Soma Membrane Voltage is Not Being Recorded")