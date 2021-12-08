# Dependencies
from neuron import h
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union
import h5py

# Project Imports
from cell_inference.utils.currents.ecp import EcpMod
from cell_inference.cells.activecell import ActiveCell
from cell_inference.cells.passivecell import PassiveCell
from cell_inference.cells.stylizedcell import CellTypes
from cell_inference.config import paths, params


class Simulation(object):
    def __init__(self, geometry: pd.DataFrame, electrodes: np.ndarray, cell_type: CellTypes,
                 loc_param: List[float] = None,
                 geo_param: Union[List[int], List[float]] = None, biophys: List[int] = None,
                 gmax: Optional[float] = None, soma_injection: Optional[np.ndarray] = None,
                 scale: float = 1.0, ncell: int = 1) -> None:
        """
        Initialize simulation object
        geometry: pandas dataframe of cell morphology properties
        electrodes: array of electrode coordinates, n-by-3
        cell_type: CellTypes enum value to indicate type of cell simulation
        loc_param: location parameters, ncell-by-6 array, (x,y,z,theta,h,phi)
        geo_param: geometry parameters, ncell-by-k array, if not specified, use default properties in geometry
        biophys: biophysical parameters, ncell-by-k array, if not specified, use default properties
        gmax: maximum conductance of synapse, ncell-vector, if this is a single value it is a constant for all cells
        soma_injection: scaling factor for passive cell soma_injections
        scale: scaling factors of lfp magnitude, ncell-vector, if is single value, is constant for all cells
        ncell: number of cells in the simulation, required if simulating for multiple cells
        """
        self.cell_type = cell_type
        self.ncell = ncell  # number of cells in this simulation
        self.cells = []  # list of cell object
        self.lfp = []  # list of EcpMod object
        self.geometry = geometry.copy()
        self.electrodes = electrodes
        self.loc_param = None
        self.geo_param = None
        self.scale = None
        self.geo_entries = [
            (0, 'R'),  # change soma radius
            (3, 'L'),  # change trunk length
            (3, 'R'),  # change trunk radius
            ([1, 2], 'R'),  # change basal dendrites radius
            (4, 'R'),  # change tuft radius
            ([1, 2, 4], 'L')  # change all dendrites length
        ]
        if loc_param is None:
            loc_param = [0., 0., 0., 0., 1., 0.]
        self.set_loc_param(loc_param)
        if geo_param is None:
            geo_param = [-1]
        self.set_geo_param(geo_param)
        self.set_scale(scale)

        self.soma_injection = None

        if cell_type == CellTypes.PASSIVE:
            if soma_injection is None:
                raise ValueError("Soma Injection is Required for a Passive Cell")
            else:
                self.soma_injection = soma_injection

        self.biophys = None
        self.gmax = None
        self.stim = None
        if cell_type == CellTypes.ACTIVE:
            if biophys is None:
                biophys = [-1]
            self.set_biophys(biophys)
            if gmax is None:
                raise ValueError("gmax is Required for an Active Cell")
            else:
                self.set_gmax(gmax)
            self.stim = self.__create_netstim()

        self.__create_cells(cell_type=cell_type)  # create cell objects with properties set up
        self.t_vec = h.Vector(round(h.tstop / h.dt) + 1).record(h._ref_t)  # record time

    @staticmethod
    def run_neuron_sim() -> None:
        """Run simulation"""
        h.run()

    # PRIVATE METHODS
    def __create_cells(self, min_dist: float = 10.0, cell_type: CellTypes = CellTypes.ACTIVE) -> None:
        """
        Create cell objects with properties set up

        Parameters
        min_dist: minimum distance allowed between segment and electrode. Set to None if not using.
        cell_type: CellTypes enum value to indicate type of cell simulation
        """
        self.cells.clear()  # remove cell objects from previous run
        self.lfp.clear()
        # Create cell with morphology and biophysical parameters
        for i in range(self.ncell):
            geometry = self.set_geometry(self.geometry, self.geo_param[i, :])
            if cell_type == CellTypes.ACTIVE:
                self.cells.append(ActiveCell(geometry=geometry, biophys=self.biophys[i, :]))
            else:
                self.cells.append(PassiveCell(geometry=geometry))
        # add injection current or synaptic current and set up lfp recording
        for i, cell in enumerate(self.cells):
            # # Pulse injection cell.add_injection(sec_index=0,record=True,delay=0.1,dur=0.2,amp=5.0) # Tune for
            # proper action potential Synpatic input
            if cell_type == CellTypes.ACTIVE:
                cell.add_synapse(self.stim, sec_index=0, gmax=self.gmax[i])
            else:
                cell.add_injection(sec_index=0, pulse=False, current=self.soma_injection, record=True)
            # Move cell location
            self.lfp.append(
                EcpMod(cell, self.electrodes, move_cell=self.loc_param[i], scale=self.scale[i], min_distance=min_dist))

    def __create_netstim(self) -> h.NetStim:
        """Setup synaptic input event"""
        stim = h.NetStim()
        stim.number = 1  # only one event
        stim.start = 0.1  # delay
        return stim

    def __pack_parameters(self, param: Optional[Union[np.ndarray, List[float], int, float]],
                          ndim: int, param_name: str) -> np.ndarray:
        """Pack parameters for the simulation"""
        if ndim == 0:
            if not hasattr(param, '__len__'):
                param = [param]
            param = np.array(param).ravel()
            if param.size != self.ncell:
                if param.size == 1:
                    param = np.broadcast_to(param, (self.ncell,))
                else:
                    raise ValueError(param_name + " size does not match ncell")
        if ndim == 1:
            param = np.array(param)
            if param.ndim == 1:
                param = np.expand_dims(param, 0)
            if param.shape[0] != self.ncell:
                if param.shape[0] == 1:
                    param = np.broadcast_to(param, (self.ncell, param.shape[1]))
                else:
                    raise ValueError(param_name + " number of rows does not match ncell")
        return param

    # PUBLIC METHODS
    def set_loc_param(self, loc_param: Optional[Union[np.ndarray, List[float]]]) -> None:
        """
        Setup location parameters.

        Parameters
        loc_param: ncell-by-6 array describing the location of the cell
        """
        loc_param = self.__pack_parameters(loc_param, 1, "loc_param")
        self.loc_param = [(loc_param[i, :3], loc_param[i, 3:]) for i in range(self.ncell)]

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

    def t(self) -> np.ndarray:
        """Return simulation time vector"""
        return self.t_vec.as_numpy()

    def get_lfp(self, index: int = 0) -> np.ndarray:
        """
        Return LFP array of the cell by index (indices), (cells-by-)channels-by-time

        Parameters
        index: index of the cell to retrieve the LFP from
        """
        if not hasattr(index, '__len__'):
            lfp = self.lfp[index].calc_ecp()
        else:
            index = np.asarray(index).ravel()
            lfp = np.stack([self.lfp[i].calc_ecp() for i in index], axis=0)
        return lfp


"""
Function to create and run a passive model simulation
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
                         cell_type=CellTypes.ACTIVE,
                         loc_param=params.LOCATION_PARAMETERS, gmax=params.GMAX)

    sim.run_neuron_sim()
    t = sim.t()
    t0 = t[:params.WINDOW_SIZE]
    return sim, params.WINDOW_SIZE, x0_trace, t0
