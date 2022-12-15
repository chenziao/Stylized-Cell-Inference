# Dependencies
from neuron import h
import pandas as pd
import numpy as np
from typing import Optional, Union, List
import warnings

# Project Imports
from cell_inference.cells.stylizedcell import StylizedCell

h.load_file('stdrun.hoc')


class ActiveFullCell(StylizedCell):
    """Define single cell model using parent class Stylized_Cell"""

    def __init__(self, geometry: Optional[pd.DataFrame] = None, biophys_type: str = 'ActiveFull',
                full_biophys: dict = None, biophys_comm: Optional[dict] = None,
                biophys: Optional[np.ndarray] = None, v_init: Optional[float] = None, **kwargs):
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        full_biophys: dictionary that includes full biophysical parameters (in Allen's celltype database model format)
        biophys_comm: dictionary that specifies common biophysical parameters for all sections
        biophys: vector of biophysical parameters corresponding to "biophys_entries". Use -1 for default value.
        v_init: inital potential of all sections. Use value in "full_biophys" if not specified.
        """
        self.grp_ids = []
        self.biophys_type = biophys_type
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
        self.grp_ids = {}
        for grp_id, ids in self.grp_sec_type_ids.items():
            self.grp_ids[grp_id] = [isec for i in ids for isec in self.sec_id_lookup[i]]
        biophys = self.default_biophys.copy()
        if self.biophys is not None:
            for i in range(len(self.biophys)):
                if not np.isnan(self.biophys[i]):
                    biophys[i] = self.biophys[i]
        self.biophys = biophys

    # PUBLIC METHODS
    def biophysical_division(self):
        """Define properties related to morphology"""
        for key, value in BIOPHYSICAL_DIVISION[self.biophys_type].items():
            setattr(self, key, value)

    def get_grp_ids(self, index) -> List:
        """Get section ids in groups(s) by index(indices) in the section group list"""
        if hasattr(index, '__len__'):
            sec_ids = [isec for i in index for isec in self.grp_ids[i]]
        else:
            sec_ids = self.grp_ids[index]
        return sec_ids

    def set_channels(self):
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
        bio_sec_ids = {name: [isec for i in ids for isec in self.sec_id_lookup[i]] for name, ids in self.section_map.items()}
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


BIOPHYSICAL_DIVISION = {
    'ActiveFull': {
        # map from biophysic section name to secion id in geometry, used with "full_biophys"
        'section_map': {'soma': [0], 'dend': [1, 2], 'apic': [3, 4], 'axon': [5]},
        # select section id's for each group, used with "biophys"
        'grp_sec_type_ids': {0: [0], 1: [1, 2], 2: [3, 4], 3: [5]},
        'biophys_entries': [(0, 'g_pas'), (1, 'g_pas'), (2, 'g_pas'), (3, 'g_pas')],
        'default_biophys': np.array([0.00051532, 0.000170972, 0.004506, 0.00951182])
    },
    'ActiveOblique': {
        'section_map': {'soma': [0], 'dend': [1, 2], 'apic': [3, 4, 5, 7], 'axon': [6]},
        'grp_sec_type_ids': {0: [0], 1: [1, 2], 2: [3, 7], 3: [4, 5], 4: [6]},
        'biophys_entries': [(1, 'Ra'), (2, 'Ra'), (3, 'Ra')],
        'default_biophys': np.array([100, 100, 100])
    },
    'ReducedOrderL5': {
        'section_map': {'soma': [0], 'dend': [1,2,3,4], 'apic': [6,7,8,9,10], 'axon': [], 'pas_dend': [12]},
        'grp_sec_type_ids': { # select section id's for each group
            0: [0], # soma
            1: [1, 2, 3], # basal group: prox, mid, dist
            2: [4], # prox trunk; 5: oblique (removed)
            3: [6], # mid trunk
            4: [7], # distal trunk (nexus)
            5: [8], # tuft: prox
            6: [9, 10], # tuft: mid, dist
            7: [12] # passive basal
        },
        'biophys_entries': [
            ([5, 6], 'e_pas'), ([5, 6], 'g_pas'),
            (0, 'gNaTa_tbar_NaTa_t'), (1, 'gNaTa_tbar_NaTa_t'),
            (0, 'gSKv3_1bar_SKv3_1'), (1, 'gSKv3_1bar_SKv3_1'),
            (1, 'Ra'), (2, 'Ra'),
            (3, 'g_pas'), (5, 'gCa_HVAbar_Ca_HVA'), (5, 'gCa_LVAstbar_Ca_LVAst'),
            (3, 'gIhbar_Ih'), (4, 'gIhbar_Ih'), (5, 'gIhbar_Ih'), (6, 'gIhbar_Ih')
        ],
        'default_biophys': np.array([
            -72.0, 0.0000589,
            2.04, 0.0639,
            0.693, 0.000261,
            100., 100.,
            0.0000525, 0.000555, 0.0187,
            0.00181, 0.00571, 0.00783, 0.01166 
        ])
    },
    'ReducedOrderL5Passive': {
        'section_map': {'soma': [0], 'dend': [1,2,3,4], 'apic': [6,7,8,9,10], 'axon': [], 'pas_dend': [12]},
        'grp_sec_type_ids': { # select section id's for each group
            0: [0], # soma
            1: [1, 2, 3], # basal group: prox,mid,dist;
            2: [4], # prox trunk; 5: oblique (removed)
            3: [6], # mid trunk
            4: [7], # distal trunk (nexus)
            5: [8], # tuft: prox
            6: [9, 10], # tuft: mid, dist
            7: [12] # passive basal
        },
        'biophys_entries': [([5, 6], 'e_pas'), ([5, 6], 'g_pas'), (1, 'Ra'), (2, 'Ra'), (3, 'g_pas')],
        'default_biophys': np.array([-72.0, 0.0000589, 100., 100., 0.0000525])
    },
    'ReducedOrderL5Stochastic': {
        'section_map': {'soma': [0], 'dend': [1,2,3,4], 'apic': [6,7,8,9,10], 'axon': [], 'pas_dend': [12]},
        'grp_sec_type_ids': { # select section id's for each group}
            0: [0], # soma
            1: [1, 2, 3], # basal group: prox, mid, dist
            2: [4], # prox trunk; 5: oblique (removed)
            3: [6], # mid trunk
            4: [7], # distal trunk (nexus)
            5: [8], # tuft: prox
            6: [9, 10], # tuft: mid, dist
            7: [12] # passive basal
        },
        'biophys_entries': [
            (0, 'gNaTa_tbar_NaTa_t'), ([1, 2], 'gNaTa_tbar_NaTa_t'), ([3, 4], 'gNaTa_tbar_NaTa_t'),
            (0, 'gSKv3_1bar_SKv3_1'), (1, 'gSKv3_1bar_SKv3_1'),
            (1, 'Ra'), (2, 'Ra'),
            (3, 'g_pas'), (5, 'gCa_HVAbar_Ca_HVA'), (5, 'gCa_LVAstbar_Ca_LVAst'),
            (3, 'gIhbar_Ih'), (4, 'gIhbar_Ih'), (5, 'gIhbar_Ih'), (6, 'gIhbar_Ih')
        ],
        'default_biophys': np.array([
            2.04, 0.0213, 0.0213,
            0.693, 0.000261,
            100., 100.,
            0.0000525, 0.000555, 0.0187,
            0.00181, 0.00571, 0.00783, 0.01166 
        ])
    }
}