import unittest

import h5py
import os

from cell_inference.utils.feature_extractors.SummaryStats2D import *
from cell_inference.config import paths


class TestSummaryStats2D(unittest.TestCase):
    def test_buildlfpgrid(self):
        lfp = np.random.uniform(low=-0.05, high=0.05, size=(100, 100))
        coord = np.random.uniform(low=0.0, high=4000.0, size=(100, 3))
        grid_v = np.random.uniform(low=0.0, high=2000.0, size=(100, 3))

        t = lfp.shape[0]
        xy = coord[:, :2]
        xx, yy = np.meshgrid(grid_v[0], grid_v[1], indexing='ij')
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        grid_lfp = np.empty((t, grid.shape[0]))
        for i in range(t):
            grid_lfp[i, :] = griddata(xy, lfp[i, :], grid)

        result_gridlfp, result_grid = build_lfp_grid(lfp, coord, grid_v)
        self.assertFalse((grid_lfp - result_gridlfp).all())
        self.assertFalse((grid - result_grid).all())

    def test_calculuatestats(self):
        hf = h5py.File(os.path.join(paths.ROOT_DIR, 'resources/cell360LFP2D.h5'), 'r')
        groundtruth_lfp = np.array(hf.get('data'))
        elec_pos = np.array(hf.get('coord'))
        elec_pos = np.column_stack((elec_pos, np.zeros(elec_pos.shape[0])))
        grid_v = (np.array(hf.get('grid/x')), np.array(hf.get('grid/y')), np.zeros(1))
        grid_lfp, grid = build_lfp_grid(groundtruth_lfp, elec_pos, grid_v)
        stats = calculate_stats(grid_lfp)

        expected = np.array([-0.32513139, 0.59316778, 3., 3., 1.46395758,
                             3., 81., -3.69506715, -8.58815789, 43.58440229,
                             1., 41., 97., 3., 22., -117., 1.05980038,
                             2.75387877, 3.,  23.02582597, 3.708033,
                             9.97216488, 3., 85.97109375, 1.19038716,
                             2.34133695, 3., 19.83984375])

        self.assertFalse((stats-expected).all())

    def test_catoutput(self):
        hf = h5py.File(os.path.join(paths.ROOT_DIR, 'resources/cell360LFP2D.h5'), 'r')
        lfp = np.array(hf.get('data'))
        include_sumstats = True

        g_lfp, grid = build_lfp_grid(lfp, params.ELECTRODE_POSITION, params.ELECTRODE_GRID)
        expected = np.concatenate((g_lfp.ravel(), calculate_stats(g_lfp, grid))) if include_sumstats else lfp.ravel()
        expected = torch.from_numpy(expected)
        output = cat_output(lfp, True)

        self.assertFalse((output-expected).all())


if __name__ == '__main__':
    unittest.main()
