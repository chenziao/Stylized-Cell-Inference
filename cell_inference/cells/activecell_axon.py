# Dependencies
from neuron import h
import pandas as pd
import numpy as np
from typing import Optional, Union
import warnings

# Project Imports
from cell_inference.cells.stylizedcell import StylizedCell

h.load_file('stdrun.hoc')


class ActiveFullCell(StylizedCell):
    """Define single cell model using parent class Stylized_Cell"""

    def __init__(self, geometry: Optional[pd.DataFrame] = None, full_biophys: dict = None,
                 biophys_comm: Optional[dict] = None, biophys: Optional[np.ndarray] = None,
                 v_init: Optional[float] = None, **kwargs) -> None:
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        full_biophys: dictionary that includes full biophysical parameters (in Allen's celltype database model format)
        biophys_comm: dictionary that specifies common biophysical parameters for all sections
        biophys: vector of biophysical parameters corresponding to "biophys_entries". Use -1 for default value.
        v_init: inital potential of all sections. Use value in "full_biophys" if not specified.
        """
        self.grp_ids = []
        self.full_biophys = full_biophys
        self.biophys = biophys
        self.biophys_comm = {} if biophys_comm is None else biophys_comm
        self.v_init = v_init

        super().__init__(geometry, **kwargs)

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
    def morphological_properties(self):
        """Define properties related to morphology"""
        # map from biophysic section name to secion id in geometry, used with "full_biophys"
        self.section_map = {'soma': [0], 'dend': [1, 2], 'apic': [3, 4], 'axon': [5]}
        # select section id's for each group, used with "biophys"
        self.grp_sec_type_ids = [[0], [1, 2], [3, 4], [5]]
        self.biophys_entries = [
            (0, 'g_pas'), (1, 'g_pas'), (2, 'g_pas'), (3, 'g_pas')
        ]
        self.default_biophys = np.array([0.00051532, 0.000170972, 0.004506, 0.00951182])
    
    def get_grp_ids(self, index):
        """Get section ids in groups(s) by index(indices) in the section group list"""
        if not hasattr(index, '__len__'):
            sec_ids = self.grp_ids[index]
        else:
            sec_ids = [isec for i in index for isec in self.grp_ids[i]]
        return sec_ids

    def set_channels(self) -> None:
        if self.full_biophys is None:
            raise ValueError("Warning: full_biophys is not loaded.")
        if set([x for xs in self.section_map.values() for x in xs])!=set(self.geometry.index.tolist()):
            print("Warning: Sections in 'section_map' are not consistent with 'geometry'.")
            print(set([x for xs in self.section_map.values() for x in xs]))
            print(set(self.geometry.index.tolist()))
        fb = self.full_biophys
        # common parameters
        if self.v_init is None: self.v_init = fb['conditions'][0]['v_init']
        h.celsius = fb['conditions'][0]['celsius']
        for sec in self.all:
            sec.Ra = fb['passive'][0]['ra']
            sec.insert('pas')
#             sec.e_pas = self._vrest
        # section specific parameters
        bio_sec_ids = {name:[isec for i in ids for isec in self.sec_id_lookup[i]] for name, ids in self.section_map.items()}
        self.bio_sec_ids = bio_sec_ids
        for genome in fb['genome']:
            mech = genome['mechanism']
            insert = mech != ""
            varname = genome['name']
            set_value = varname != "" and genome['value'] != ""
            for isec in bio_sec_ids[genome['section']]:
                sec = self.get_sec_by_id(isec)
                if insert:
                    sec.insert(mech)
                if set_value:
                    setattr(sec, varname, genome['value'])
        for erev in fb['conditions'][0]['erev']:
            enames = [x for x in erev.keys() if x != 'section']
            for isec in bio_sec_ids[erev['section']]:
                sec = self.get_sec_by_id(isec)
                for en in enames:
                    setattr(sec, en, erev[en])
        # modified common parameters
        for key, value in self.biophys_comm.items():
            for sec in self.all:
                setattr(sec, key, value)
        
        # variable parameters
        if not self.grp_ids:
            self.__create_biophys_entries()
        for i, entry in enumerate(self.biophys_entries):
            for sec in self.get_sec_by_id(self.get_grp_ids(entry[0])):
                try:
                    setattr(sec, entry[1], self.biophys[i])
                except AttributeError:
                    warnings.warn("Error: {} not found in {}".format(entry[1], sec))

        h.v_init = self.v_init


class ActiveObliqueCell(ActiveFullCell):
    """Define single cell model using parent class ActiveFullCell"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def morphological_properties(self):
        """Define properties related to morphology"""
        # map from biophysic section name to secion id in geometry
        self.section_map = {'soma': [0], 'dend': [1, 2], 'apic': [3, 4, 5, 7], 'axon': [6]}
        # select section id's for each group
        self.grp_sec_type_ids = [[0], [1, 2], [3, 7], [4, 5], [6]] # soma, basal, trunk, tuft, axon
        self.biophys_entries = [
            (1, 'Ra'), (2, 'Ra'), (3, 'Ra')
        ]
        self.default_biophys = np.array([100, 100, 100])


class ReducedOrderL5Cell(ActiveFullCell):
    """Reduced order cell with active dendrites"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def morphological_properties(self):
        """Define properties related to morphology"""
        # map from biophysic section name to secion id in geometry, used with "full_biophys"
        self.section_map = {'soma': [0], 'dend': [1,2,3,4], 'apic': [6,7,8,9,10], 'axon': [], 'pas_dend': [12]}
        # select section id's for each group, used with "biophys"
        self.grp_sec_type_ids = [ # select section id's for each group
                                 [0], # soma
                                 [1, 2, 3], # basal group: prox, mid, dist
                                 [4], # prox trunk; 5: oblique (removed)
                                 [6], # mid trunk
                                 [7], # distal trunk (nexus)
                                 [8], # tuft: prox
                                 [9, 10], # tuft: mid, dist
                                 [12] # passive basal
                                ]
        self.biophys_entries = [
            ([5, 6], 'e_pas'), ([5, 6], 'g_pas'),
            (0, 'gNaTa_tbar_NaTa_t'), (1, 'gNaTa_tbar_NaTa_t'),
            (0, 'gSKv3_1bar_SKv3_1'), (1, 'gSKv3_1bar_SKv3_1'),
            (1, 'Ra'), (2, 'Ra'),
            (3, 'g_pas'), ([4, 5], 'gCa_HVAbar_Ca_HVA'), ([4, 5], 'gCa_LVAstbar_Ca_LVAst'),
            (3, 'gIhbar_Ih'), (4, 'gIhbar_Ih'), (5, 'gIhbar_Ih'), (6, 'gIhbar_Ih')
        ]
        self.default_biophys = np.array([
            -72.0, 0.0000589,
            2.04, 0.0639,
            0.693, 0.000261,
            100., 100.,
            0.0000525, 0.000555, 0.0187,
            0.00181, 0.00571, 0.00783, 0.01166 
        ])

class ReducedOrderL5CellPassive(ActiveFullCell):
    """Reduced order cell with passive dendrites"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def morphological_properties(self):
        """Define properties related to morphology"""
        # map from biophysic section name to secion id in geometry, used with "full_biophys"
        self.section_map = {'soma': [0], 'dend': [1,2,3,4], 'apic': [6,7,8,9,10], 'axon': [], 'pas_dend': [12]}
        # select section id's for each group, used with "biophys"
        self.grp_sec_type_ids = [ # select section id's for each group
                                 [0], # soma
                                 [1, 2, 3], # basal group: prox,mid,dist;
                                 [4], # prox trunk; 5: oblique (removed)
                                 [6], # mid trunk
                                 [7], # distal trunk (nexus)
                                 [8], # tuft: prox
                                 [9, 10], # tuft: mid, dist
                                 [12] # passive basal
                                ]
        self.biophys_entries = [
            ([5, 6], 'e_pas'), ([5, 6], 'g_pas'), (1, 'Ra'), (2, 'Ra'), (3, 'g_pas')
        ]
        self.default_biophys = np.array([-72.0, 0.0000589, 100., 100., 0.0000525])
