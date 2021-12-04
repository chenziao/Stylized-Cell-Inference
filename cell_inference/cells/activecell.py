# Dependencies
from neuron import h
import pandas as pd
import numpy as np
from typing import Optional, Any, Union

# Project Imports
from cell_inference.cells.stylizedcell import StylizedCell
from cell_inference.cells.synapse import Synapse
from cell_inference.utils.currents.recorder import Recorder

h.load_file('stdrun.hoc')


class ActiveCell(StylizedCell):
    """Define single cell model using parent class Stylized_Cell"""

    def __init__(self, geometry: Optional[pd.DataFrame] = None,
                 biophys: Optional[np.ndarray] = None,
                 dl: int = 30, vrest: float = -70.0) -> None:
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
        self.biophys_entries = [
            (0, 'g_pas'), (1, 'g_pas'), (2, 'g_pas'),  # g_pas of soma, basal, apical
            (0, 'gNaTa_tbar_NaTa_t'), (2, 'gNaTa_tbar_NaTa_t'),  # gNaTa_t of soma, apical
            (0, 'gSKv3_1bar_SKv3_1'), (2, 'gSKv3_1bar_SKv3_1')  # gSKv3_1 of soma, apical
        ]
        
        super(ActiveCell, self).__init__(geometry, dl, vrest)
        self.v_rec = self.__record_soma_v()
        
#         self.set_channels()

    # PRIVATE METHODS
    def __create_biophys_entries(self, biophys: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Define list of entries of biophysical parameters.
        Each entry is a pair of group id and parameter reference string.
        Define default values and set parameters in "biophys".
        """
        grp_sec_type_ids = [[0], [1, 2], [3, 4]]  # select section id's for each group
        for ids in grp_sec_type_ids:
            secs = []
            for i in ids:
                secs.extend(self.sec_id_lookup[i])
            self.grp_ids.append(secs)
        default_biophys = np.array([0.0000338, 0.0000467, 0.0000589, 2.04, 0.0213, 0.693, 0.000261])
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
        soma.insert('NaTa_t')  # Sodium channel
        soma.insert('SKv3_1')  # Potassium channel
        soma.ena = 50
        soma.ek = -85
        for isec in self.grp_ids[2]:
            sec = self.get_sec_by_id(isec)  # apical dendrites
            if not hasattr(sec, '__len__'):
                for s in sec:
                    s.insert('NaTa_t')
                    s.insert('SKv3_1')
                    s.ena = 50
                    s.ek = -85
            else:
                sec.insert('NaTa_t')
                sec.insert('SKv3_1')
                sec.ena = 50
                sec.ek = -85
        # variable parameters
        for i, entry in enumerate(self.biophys_entries):
            for sec in self.get_sec_by_id(self.grp_ids[entry[0]]):
                setattr(sec, entry[1], self.biophys[i])
        h.v_init = self._vrest

    def add_synapse(self, stim: h.NetStim, sec_index: int, **kwargs: Any) -> None:
        """Add synapse to a section by its index"""
        self.injection.append(Synapse(self, stim, sec_index, **kwargs))

    def v(self) -> Optional[Union[str, np.ndarray]]:
        """Return recorded soma membrane voltage in numpy array"""
        if hasattr(self, 'v_rec'):
            return self.v_rec.as_numpy()
        else:
            raise NotImplemented("Soma Membrane Voltage is Not Being Recorded")
