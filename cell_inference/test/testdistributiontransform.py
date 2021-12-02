import unittest

import numpy as np

from cell_inference.utils.transform.distribution_transformation import *


class TestDistributionTransform(unittest.TestCase):

    def test_norm2unif(self):
        x, a, b = 1, 4, 5
        y = a + (b - a) * norm.cdf(np.asarray(x))
        self.assertEqual(norm2unif(x, a, b), y)  # Test ints
        self.assertEqual(norm2unif(float(x), float(a), float(b)), y)  # Test floats
        self.assertEqual(norm2unif(np.asarray(x), np.asarray(a), np.asarray(b)), y)  # Test np.ndarrays

    def test_range2logn(self):
        a, b, n_sigma = 200, 400, 3
        mu = (np.log(a) + np.log(b)) / 2
        sigma = (np.log(b) - np.log(a)) / n_sigma / 2

        # INT CASE
        y1, y2 = range2logn(a, b, n_sigma)
        self.assertEqual(mu, y1)
        self.assertEqual(sigma, y2)

        # FLOAT CASE
        y1, y2 = range2logn(float(a), float(b), float(n_sigma))
        self.assertEqual(mu, y1)
        self.assertEqual(sigma, y2)

        # NP.NDARRAY CASE
        y1, y2 = range2logn(np.asarray(a), np.asarray(b), np.asarray(n_sigma))
        self.assertEqual(mu, y1)
        self.assertEqual(sigma, y2)

    # TODO Numpy tests fail for the following 3 conditions. Need to be fixed and accounted for
    def test_norm2logn(self):
        x, mu, sigma = 5, 1, 1
        y = np.exp(mu + sigma * x)
        self.assertEqual(norm2logn(x, mu, sigma), y)
        self.assertEqual(norm2logn(float(x), float(mu), float(sigma)), y)
        # self.assertEqual(norm2logn(np.ndarray(x), np.ndarray(mu), np.ndarray(sigma)), y)

    def test_logds_norm2unif(self):
        x, a, b = 2, 1, 3
        logds = -np.log(b - a) - np.log(norm.pdf(x))
        self.assertEqual(logds_norm2unif(x, a, b), logds)
        self.assertEqual(logds_norm2unif(float(x), float(a), float(b)), logds)
        # self.assertEqual(logds_norm2unif(np.ndarray(x), np.ndarray(a), np.ndarray(b)), logds)

    def test_logds_norm2logn(self):
        x, mu, sigma = 3, 2, 5
        logds = -np.log(sigma) - mu - sigma * x
        self.assertEqual(logds_norm2logn(x, mu, sigma), logds)
        self.assertEqual(logds_norm2logn(float(x), float(mu), sigma), logds)
        # self.assertEqual(logds_norm2logn(np.ndarray(x), np.ndarray(mu), sigma), logds)


if __name__ == '__main__':
    unittest.main()
