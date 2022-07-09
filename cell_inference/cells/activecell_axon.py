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


class ActiveAxonCell(StylizedCell):
    """Define single cell model using parent class Stylized_Cell"""

    def __init__(self, geometry: Optional[pd.DataFrame] = None, full_biophys: dict = None,
                 biophys: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        full_biophys: dictionary that includes full biophysical parameters (in Allen's celltype database model format)
        biophys: vector of biophysical parameters corresponding to "biophys_entries". Use -1 for default value.
        dL: maximum segment length
        vrest: reversal potential for leak channels
        """
        self.full_biophys = full_biophys
        self.biophys = biophys
        self.section_map = {'soma':[0],'dend':[1,2],'apic':[3,4],'axon':[5]}  # map from biophysic section name to secion id in geometry
        self.grp_sec_type_ids = [[0], [1, 2], [3, 4], [5]]  # select section id's for each group
        self.biophys_entries = [
            (0, 'g_pas'), (1, 'g_pas'), (2, 'g_pas'), (3, 'g_pas')
        ]
        self.grp_ids = []
        
        super().__init__(geometry, **kwargs)
        self.v_rec = self.__record_soma_v()
        
#         self.set_channels()

    # PRIVATE METHODS
    def __create_biophys_entries(self, biophys: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Define list of entries of biophysical parameters.
        Each entry is a pair of group id and parameter reference string.
        Define default values and set parameters in "biophys".
        """
        self.grp_ids = [[isec for i in ids for isec in self.sec_id_lookup[i]] for ids in self.grp_sec_type_ids]
        default_biophys = np.array([0.00051532, 0.000170972, 0.004506, 0.00951182])
        if biophys is not None:
            for i in range(len(biophys)):
                if biophys[i] >= 0:
                    default_biophys[i] = biophys[i]
        return default_biophys

    def __record_soma_v(self) -> Recorder:
        return Recorder(self.soma(.5), 'v')

    # PUBLIC METHODS
    def set_channels(self) -> None:
        if self.full_biophys is None:
            raise ValueError("Warning: full_biophys is not loaded.")
        if set([x for xs in self.section_map.values() for x in xs])!=set(self.geometry.index.to_list()):
            print("Warning: Sections in 'section_map' are not consistent with 'geometry'.")
        fb = self.full_biophys
        # common parameters
        self._vrest = fb['conditions'][0]['v_init']
        h.celsius = fb['conditions'][0]['celsius']
        for sec in self.all:
            sec.Ra = fb['passive'][0]['ra']
            sec.insert('pas')
#             sec.e_pas = self._vrest
#             sec.cm = 2.0
        # section specific parameters
#         bio_sec_ids = {}
#         for name, ids in self.section_map.items():
#             bio_sec_ids[name] = []
#             for i in ids:
#                 bio_sec_ids[name].extend(self.sec_id_lookup[i])
        bio_sec_ids = {name:[isec for i in ids for isec in self.sec_id_lookup[i]] for name, ids in self.section_map.items()}
        self.bio_sec_ids = bio_sec_ids
        for genome in fb['genome']:
            mech = genome['mechanism']
            insert = mech != ""
            for isec in bio_sec_ids[genome['section']]:
                sec = self.get_sec_by_id(isec)
                if insert:
                    sec.insert(mech)
                setattr(sec, genome['name'], float(genome['value']))
        for erev in fb['conditions'][0]['erev']:
            enames = [x for x in erev.keys() if x != 'section']
            for isec in bio_sec_ids[erev['section']]:
                sec = self.get_sec_by_id(isec)
                for en in enames:
                    setattr(sec, en, erev[en])
        
        # variable parameters
        if not self.grp_ids:
            self.biophys = self.__create_biophys_entries(self.biophys)
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
