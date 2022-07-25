# Dependencies
from neuron import h
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union
import h5py

# Project Imports
from cell_inference.cells.stylizedcell import CellTypes
from cell_inference.utils.currents.ecp import EcpMod
from cell_inference.config import paths, params


class Simulation(object):
    def __init__(self, geometry: pd.DataFrame, full_biophys: Optional[dict] = None,
                 cell_type: CellTypes = CellTypes.ACTIVE,
                 electrodes: Optional[np.ndarray] = None,
                 loc_param: Union[np.ndarray, List[int], List[float]] = None,
                 geo_param: Union[np.ndarray, List[int], List[float]] = None,
                 biophys: Union[np.ndarray, List[int], List[float]] = None,
                 biophys_comm: Optional[dict] = None,
                 spike_threshold: Optional[float] = None,
                 gmax: Optional[float] = None, stim_param: Optional[dict] = {},
                 soma_injection: Optional[np.ndarray] = None,
                 scale: float = 1.0, ncell: int = 1) -> None:
        """
        Initialize simulation object
        geometry: pandas dataframe of cell morphology properties
        electrodes: array of electrode coordinates, n-by-3
        cell_type: CellTypes enum value to indicate type of cell simulation
        loc_param: location parameters, ncell-by-6 (or ncell-by-nloc-by-6) array, (x,y,z,theta,h,phi)
        geo_param: geometry parameters, ncell-by-k array, if not specified, use default properties in geometry
        biophys: biophysical parameters, ncell-by-k array, if not specified, use default properties
        spike_threshold: membrane voltage threshold for recording spikes, if not specified, do not record
        gmax: maximum conductance of synapse, ncell-vector, if this is a single value it is a constant for all cells'
        soma_injection: scaling factor for passive cell soma_injections
        scale: scaling factors of lfp magnitude, ncell-vector, if is single value, is constant for all cells
        ncell: number of cells in the simulation, required if simulating for multiple cells
        """
        self.cell_type = cell_type
        self.ncell = ncell  # number of cells in this simulation
        self.cells = []  # list of cell object
        self.electrodes = np.array(electrodes)
        self.lfp = []  # list of EcpMod object
        self.min_dist = 10.0 # minimum distance allowed between segment and electrode. Set to None if not using.
        self.loc_param = None
        if loc_param is None:
            loc_param = [0., 0., 0., 0., 1., 0.]
        self.set_loc_param(loc_param)
        self.geometry = geometry.copy()
        self.geo_entries = [
            (0, 'R'),  # change soma radius
            (3, 'L'),  # change trunk length
            (3, 'R'),  # change trunk radius
            ([1, 2], 'R'),  # change basal dendrites radius
            (4, 'R'),  # change tuft radius
            ([1, 2, 4], 'L')  # change all dendrites length
        ]
        self.geo_param = None
        if geo_param is None:
            geo_param = []
        self.set_geo_param(geo_param)
        self.scale = None
        self.set_scale(scale)

        self.soma_injection = None
        if cell_type == CellTypes.PASSIVE:
            if soma_injection is None:
                raise ValueError("Soma Injection is Required for a Passive Cell")
            else:
                self.soma_injection = soma_injection

        self.full_biophys = None
        self.biophys = None
        self.biophys_comm = {}
        self.gmax = None
        self.stim = None
        if cell_type != CellTypes.PASSIVE:
            if full_biophys is not None:
                self.full_biophys = full_biophys
                for genome in self.full_biophys['genome']:
                    genome['value'] = float(genome['value'])
            if biophys is None:
                biophys = []
            if biophys_comm is not None:
                self.biophys_comm = biophys_comm
            self.set_biophys(biophys)
            if gmax is None:
                print("Warning: Not using synaptic input. gmax is required for synaptic input in an Active Cell.")
            else:
                self.set_gmax(gmax)
                self.stim = self.__create_netstim(stim_param)
            if cell_type == CellTypes.ACTIVE_AXON:
               self.geo_entries.append((5, 'R'))

        self.__load_cell_module()
        self.__create_cells()  # create cell objects with properties set up
        self.t_vec = h.Vector(round(h.tstop / h.dt) + 1).record(h._ref_t)  # record time
        self.spike_threshold = None
        self.record_spikes(spike_threshold)

    @staticmethod
    def run_neuron_sim() -> None:
        """Run simulation"""
        h.run()

    # PRIVATE METHODS
    def __load_cell_module(self,) -> None:
        """Load cell module and define arguments for initializing cell instance according to cell type"""
        def pass_geometry(CellClass):
            def create_cell(i,**kwargs):
                return CellClass(geometry=self.set_geometry(self.geometry, self.geo_param[i, :]),**kwargs)
            return create_cell
        if self.cell_type == CellTypes.PASSIVE:
            from cell_inference.cells.passivecell import PassiveCell
            self.CreateCell = pass_geometry(PassiveCell)
        if self.cell_type == CellTypes.ACTIVE:
            from cell_inference.cells.activecell import ActiveCell
            create_cell = pass_geometry(ActiveCell)
            self.CreateCell = lambda i: create_cell(i,biophys=self.biophys[i, :])
        if self.cell_type == CellTypes.ACTIVE_AXON:
            from cell_inference.cells.activecell_axon import ActiveAxonCell
            create_cell = pass_geometry(ActiveAxonCell)
            self.CreateCell = lambda i: create_cell(i,biophys=self.biophys[i, :],
                full_biophys=self.full_biophys,biophys_comm=self.biophys_comm)
    
    def __create_cells(self) -> None:
        """Create cell objects with properties set up"""
        self.cells.clear()  # remove cell objects from previous run
        self.lfp.clear()
        # Create cell with morphology and biophysical parameters
        for i in range(self.ncell):
            self.cells.append(self.CreateCell(i))
        # add injection current or synaptic current and set up lfp recording
        for i, cell in enumerate(self.cells):
            if self.cell_type != CellTypes.PASSIVE:
                if self.gmax is not None:
                    cell.add_synapse(self.stim, sec_index=0, gmax=self.gmax[i])
            else:
                cell.add_injection(sec_index=0, pulse=False, current=self.soma_injection, record=True)
            # Move cell location
            if self.electrodes is not None:
                self.lfp.append(
                    EcpMod(cell, self.electrodes, move_cell=self.loc_param[i,0], scale=self.scale[i], min_distance=self.min_dist))

    def __create_netstim(self, stim_param: Optional[dict] = {}) -> h.NetStim:
        """Setup synaptic input event"""
        stim = h.NetStim()
        stim.number = 1  # only one event
        stim.start = 0.1  # delay
        for key, value in stim_param.items():
            setattr(stim, key, value)
        return stim

    def __pack_parameters(self, param: Optional[Union[np.ndarray, List[float], int, float]],
                          ndim: int, param_name: str) -> np.ndarray:
        """
        Pack parameters for the simulation into numpy arrays with cells along the first dimension.
        
        Parameters
        param: (array of) input parameter(s)
        ndim: number of dimension of parameters for each cell
        param_name: parameter set name for printing error message
        """
        param = np.asarray(param)
        if param.ndim > ndim + 1:
            raise ValueError("%s has more dimensions size than expected" % param_name)
        # add dimensions before the trailing dimension
        param = np.expand_dims(param,tuple(range(param.ndim-1,ndim)))
        if param.shape[0] != self.ncell:
            if param.shape[0] == 1:
                # repeat the same parameters for all cells
                param = np.broadcast_to(param, (self.ncell,) + param.shape[1:])
            else:
                raise ValueError("%s first dimension size does not match ncell=%d" % (param_name,self.ncell))
        return param

    # PUBLIC METHODS
    def set_loc_param(self, loc_param: Optional[Union[np.ndarray, List[float]]]) -> None:
        """
        Setup location parameters.

        Parameters
        loc_param: ncell-by-6 (or ncell-by-nloc-by-6) array describing the location of the cell
        """
        loc_param = self.__pack_parameters(loc_param, 2, "loc_param")
        # reshape the trailing dimension to 2-by-3
        self.loc_param = loc_param.reshape(loc_param.shape[:2] + (2, 3))

    def set_geo_param(self, geo_param: Optional[Union[np.ndarray, List[float]]]) -> None:
        """
        Setup geometry parameters.

        Parameters
        geo_param: ncell-by-k array, k entries of properties
        """
        self.geo_param = self.__pack_parameters(geo_param, 1, "geo_param")

    def set_biophys(self, biophys: Optional[Union[np.ndarray, List[float]]]) -> None:
        """
        Setup geometry parameters.

        Parameters
        biophys: ncell-by-k array, k entries of properties
        """
        self.biophys = self.__pack_parameters(biophys, 1, "biophys")

    def set_gmax(self, gmax: float) -> None:
        """
        Setup maximum conductance of synapse

        Parameters
        gmax: cell gmax value
        """
        self.gmax = self.__pack_parameters(gmax, 0, "gmax")
        for i, cell in enumerate(self.cells):
            cell.synapse[0].set_gmax(self.gmax[i])

    def set_scale(self, scale: float) -> None:
        """
        setup scaling factors of lfp magnitude

        Parameters
        scale: LFP scaling constant
        """
        self.scale = self.__pack_parameters(scale, 0, "scale")

    def set_geometry(self, geometry: pd.DataFrame, geo_param: np.ndarray) -> pd.DataFrame:
        """
        Set property values from geo_param through each entry to geometry.

        Parameters
        geometry: pandas dataframe describing the geometry
        geo_param: numpy array describing entries used
        """
        geom = geometry.copy()
        for i, x in enumerate(geo_param):
            if x >= 0:
                geom.loc[self.geo_entries[i]] = x
        return geom

    def record_spikes(self, threshold: Optional[float]) -> None:
        """Setup spike recorder for all cells"""
        for cell in self.cells:
            cell.set_spike_recorder(threshold)
        self.spike_threshold = threshold

    def t(self) -> np.ndarray:
        """Return simulation time vector"""
        return self.t_vec.as_numpy().copy()

    def get_lfp(self, index: Union[np.ndarray, List[int], int, str] = 0,
                multiple_position: bool = False) -> np.ndarray:
        """
        Return LFP array of the cell by index (indices), (cells-by-)channels-by-time

        Parameters
        index: index of the cell to retrieve the LFP from
        multiple_position: get from multiple positions for each cell along second dimension of the LFP array
        """
        if multiple_position:
            calc_ecp = lambda i: self.lfp[i].calc_ecps(move_cell=self.loc_param[i])
        else:
            calc_ecp = lambda i: self.lfp[i].calc_ecp()
        if index == 'all':
            index = range(self.ncell)
        if not hasattr(index, '__len__'):
            lfp = calc_ecp(index)
        else:
            index = np.asarray(index).ravel()
            lfp = np.stack([calc_ecp(i) for i in index], axis=0)
        return lfp

    def v(self, index: Union[np.ndarray, List[int], int, str] = 0) -> np.ndarray:
        """
        Return soma membrane potential of the cell by index (indices), (cells-by-)time

        Parameters
        index: index of the cell to retrieve the soma Vm from
        """
        if index == 'all':
            index = range(self.ncell)
        if not hasattr(index, '__len__'):
            v = self.cells[index].v()
        else:
            index = np.asarray(index).ravel()
            v = np.stack([self.cells[i].v() for i in index], axis=0)
        return v.copy()

    def get_spike_time(self, index: Union[np.ndarray, List[int], int, str] = 0) -> np.ndarray:
        """
        Return soma spike time of the cell by index (indices), ndarray (list of ndarray)

        Parameters
        index: index of the cell to retrieve the spikes from
        """
        if self.spike_threshold is None:
            raise ValueError("Spike recorder was not set up.")
        if type(index) is str and index == 'all':
            index = range(self.ncell)
        if not hasattr(index, '__len__'):
            spk = self.cells[index].spikes.as_numpy().copy()
        else:
            index = np.asarray(index).ravel()
            spk = np.array([self.cells[i].spikes.as_numpy().copy() for i in index], dtype=object)
        return spk

    def get_spike_number(self, index: Union[np.ndarray, List[int], int, str] = 0) -> Union[int, np.ndarray]:
        """
        Return soma spike number of the cell by index (indices), int (ndarray)

        Parameters
        index: index of the cell to retrieve the spikes from
        """
        if self.spike_threshold is None:
            raise ValueError("Spike recorder was not set up.")
        if index == 'all':
            index = range(self.ncell)
        if not hasattr(index, '__len__'):
            spk = self.get_spike_time(index)
            nspk = spk.size
        else:
            index = np.asarray(index).ravel()
            spk = self.get_spike_time(index)
            nspk = np.array([s.size for s in spk])
        return nspk, spk


"""
Function to create and run a simulation
"""

def run_simulation(cell_type: CellTypes) -> Tuple[Simulation, int, np.ndarray, np.ndarray]:
    h.nrn_load_dll(paths.COMPILED_LIBRARY)
    geo_standard = pd.read_csv(paths.GEO_STANDARD, index_col='id')
    h.tstop = params.TSTOP
    h.dt = params.DT
    hf = h5py.File(paths.SIMULATED_DATA_FILE, 'r')
    groundtruth_lfp = np.array(hf.get('data'))
    hf.close()
    x0_trace = groundtruth_lfp[params.START_IDX:params.START_IDX + params.WINDOW_SIZE, :]
    if cell_type == CellTypes.PASSIVE:
        max_indx = np.argmax(np.absolute(groundtruth_lfp).max(axis=0))
        max_trace = -groundtruth_lfp[params.START_IDX:, max_indx]
        soma_injection = np.insert(max_trace, 0, 0.)
        soma_injection = np.asarray([s * params.SOMA_INJECT_SCALING_FACTOR for s in soma_injection])
        sim = Simulation(geo_standard, params.ELECTRODE_POSITION,
                         cell_type=CellTypes.PASSIVE, soma_injection=soma_injection)
    else:
        sim = Simulation(geo_standard, params.ELECTRODE_POSITION,
                         cell_type=cell_type,
                         loc_param=params.LOCATION_PARAMETERS, gmax=params.GMAX)

    sim.run_neuron_sim()
    t = sim.t()
    t0 = t[:params.WINDOW_SIZE]
    return sim, params.WINDOW_SIZE, x0_trace, t0
