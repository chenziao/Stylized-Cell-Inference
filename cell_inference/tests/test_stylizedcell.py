import unittest

import pandas as pd
import numpy as np
import h5py
from neuron import h
from typing import Tuple

from cell_inference.cells.stylizedcell import CellTypes
from cell_inference.cells.simulation import Simulation
from cell_inference.utils.feature_extractors.SummaryStats2D import build_lfp_grid, calculate_stats
from cell_inference.config import params, paths


def calculate_error(psj: np.ndarray, asj: np.ndarray, error: int) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Helper function, takes in n of 1 summary stat from both passive and active cells

    psj: n amount of Summary Stat j from the passive cell model
    asj: n amount of Summary Stat j from the active cell model

    returns False if error > 1 or True if error < 1
    """
    numer = np.mean(psj - asj)
    denom = np.sqrt((np.var(psj) + np.var(asj)) / 2)
    return (numer / denom) < error, numer, denom


class TestStylizedCell(unittest.TestCase):
    def setUp(self) -> None:
        h.nrn_load_dll(paths.COMPILED_LIBRARY)
        h.tstop = params.TSTOP
        h.dt = params.DT

        self.ncell = 100

        self.rng = np.random.default_rng(12345)
        x = self.rng.uniform(low=-20., high=20., size=self.ncell)
        y = self.rng.uniform(low=-2000., high=2000., size=self.ncell)
        z = self.rng.uniform(low=-20., high=20., size=self.ncell)
        a_rot = self.rng.uniform(low=0., high=np.pi, size=self.ncell)
        h_rot = self.rng.uniform(low=-1., high=1., size=self.ncell)
        p_rot = self.rng.uniform(low=-0., high=np.pi, size=self.ncell)
        self.loc_param = np.stack((x, y, z, a_rot, h_rot, p_rot), axis=-1)
        self.geo_param = np.tile(np.array([6.0, 400.0, 0.5, 0.5, 0.5, 200.0]), (self.ncell, 1))
        self.geo_standard = pd.read_csv(paths.GEO_STANDARD, index_col='id')

        hf = h5py.File(paths.SIMULATED_DATA_FILE, 'r')
        groundtruth_lfp = np.array(hf.get('data'))
        hf.close()
        max_indx = np.argmax(
            np.absolute(groundtruth_lfp).max(axis=0))  # find maximum absolute value from averaged traces
        max_trace = -groundtruth_lfp[params.START_IDX:, max_indx]
        soma_injection = np.insert(max_trace, 0, 0.)

        self.soma_injection = np.asarray([s * params.SOMA_INJECT_SCALING_FACTOR for s in soma_injection])

        self.passive_sim = Simulation(geometry=self.geo_standard,
                                      electrodes=params.ELECTRODE_POSITION,
                                      cell_type=CellTypes.PASSIVE,
                                      loc_param=self.loc_param,
                                      geo_param=self.geo_param,
                                      soma_injection=self.soma_injection,
                                      ncell=self.ncell)

        self.active_sim = Simulation(geometry=self.geo_standard,
                                     electrodes=params.ELECTRODE_POSITION,
                                     cell_type=CellTypes.ACTIVE,
                                     loc_param=self.loc_param,
                                     geo_param=self.geo_param,
                                     gmax=0.005,
                                     scale=1.,
                                     ncell=self.ncell)

    def test_active_passive_stats(self):
        self.passive_sim.run_neuron_sim()
        self.active_sim.run_neuron_sim()

        passive_lfp = self.passive_sim.get_lfp(np.arange(self.ncell)).T
        active_lfp = self.active_sim.get_lfp(np.arange(self.ncell)).T

        passive_stats_list = []
        active_stats_list = []

        for cell in range(self.ncell):
            passive_g_lfp, passive_grid = build_lfp_grid(passive_lfp[:, :, cell],
                                                         params.ELECTRODE_POSITION,
                                                         params.ELECTRODE_GRID)
            active_g_lfp, active_grid = build_lfp_grid(active_lfp[:, :, cell],
                                                       params.ELECTRODE_POSITION,
                                                       params.ELECTRODE_GRID)
            passive_stats_list.append(calculate_stats(passive_g_lfp, passive_grid))
            active_stats_list.append(calculate_stats(active_g_lfp, active_grid))

        passive_stats = np.array(passive_stats_list)
        active_stats = np.array(active_stats_list)

        for j in range(passive_stats.shape[1]):
            check, numer, denom = calculate_error(passive_stats[:, j], active_stats[:, j], 1)
            self.assertTrue(check, "Summary Stat {} with num {} and denom {}".format(j, numer, denom))


if __name__ == '__main__':
    unittest.main()
