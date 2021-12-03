import unittest

from cell_inference.utils.feature_extractors.SummaryStats2D import *


class TestSummaryStats2D(unittest.TestCase):
    def test_buildlfpgrid(self):
        lfp = np.array([5., 3.][5., 2.][5., 1.])
        coord = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.]])
        grid_v = np.array([[0., 0., 1.],
                           [1., 1., 2.],
                           [2., 2., 3.],
                           [3., 3., 4.],
                           [4., 4., 5.]])

        t = lfp.shape[0]

        xy = coord[:, :2]

        xx, yy = np.meshgrid(grid_v[0], grid_v[1], indexing='ij')

        grid = np.column_stack((xx.ravel(), yy.ravel()))

        grid_lfp = np.empty((t, grid.shape[0]))

        for i in range(t):
            grid_lfp[i, :] = griddata(xy, lfp[i, :], grid)

        self.assertEqual(grid_lfp, build_lfp_grid(lfp, coord, grid_v)[0])
        self.assertEqual(grid, build_lfp_grid(lfp, coord, grid_v)[1])


if __name__ == '__main__':
    unittest.main()
