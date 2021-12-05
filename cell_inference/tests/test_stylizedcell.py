import unittest

import pandas as pd
import numpy as np
import h5py
from neuron import h

from cell_inference.cells.stylizedcell import CellTypes
from cell_inference.cells.simulation import Simulation
from cell_inference.utils.feature_extractors.SummaryStats2D import build_lfp_grid, calculate_stats
from cell_inference.config import params, paths


class TestStylizedCell(unittest.TestCase):
    def setUp(self) -> None:
        h.nrn_load_dll(paths.COMPILED_LIBRARY)
        h.tstop = params.TSTOP
        h.dt = params.DT

        self.rng = np.random.default_rng(12345)
        self.loc_param = [0, 0, 50, 0, 1.0, 0.0]
        self.geo_param = [6.0, 400.0, 0.5, 0.5, 0.5, 200.0]
        self.geo_standard = pd.read_csv(paths.GEO_STANDARD, index_col='id')

        hf = h5py.File(paths.INVIVO_DATA_FILE, 'r')
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
                                      ncell=1)

        self.active_sim = Simulation(geometry=self.geo_standard,
                                     electrodes=params.ELECTRODE_POSITION,
                                     cell_type=CellTypes.ACTIVE,
                                     loc_param=self.loc_param,
                                     geo_param=self.geo_param,
                                     gmax=0.005,
                                     scale=1.,
                                     ncell=1)

    def test_active_passive_stats(self):
        self.passive_sim.run_neuron_sim()
        self.active_sim.run_neuron_sim()

        passive_lfp = self.passive_sim.get_lfp().T
        active_lfp = self.active_sim.get_lfp().T

        passive_g_lfp, passive_grid = build_lfp_grid(passive_lfp, params.ELECTRODE_POSITION, params.ELECTRODE_GRID)
        active_g_lfp, active_grid = build_lfp_grid(active_lfp, params.ELECTRODE_POSITION, params.ELECTRODE_GRID)
        passive_stats = calculate_stats(passive_g_lfp, passive_grid)
        active_stats = calculate_stats(active_g_lfp, active_grid)

        np.testing.assert_array_almost_equal(passive_stats, active_stats)


if __name__ == '__main__':
    unittest.main()
