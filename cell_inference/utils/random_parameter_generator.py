import numpy as np
from typing import Optional, List, Tuple, Dict
from cell_inference.utils.transform.distribution_transformation import range2norm, range2logn

class Random_Parameter_Generator(object):
    def __init__(self, seed: Optional[int] = None, n_sigma: float = 3.0 ):
        """ Random simulation parameter generator """
        self.rng = np.random.default_rng(seed)
        self.n_sigma = n_sigma

    # DEFINE GENERATOR FOR EACH DISTRIBUTION TYPE
    # uniform distribution
    def unif(self, p_range, size=None):
        return self.rng.uniform(low=p_range[0], high=p_range[1], size=size)

    # normal distribution
    def norm(self, p_range, size=None):
        mu, sigma = range2norm(p_range[0], p_range[1], n_sigma=self.n_sigma)
        return self.rng.normal(loc=mu, scale=sigma, size=size)

    # lognormal distribution
    def logn(self, p_range, size=None):
        mu, sigma = range2logn(p_range[0], p_range[1], n_sigma=self.n_sigma)
        return self.rng.lognormal(mean=mu, sigma=sigma, size=size)

    # METHODS
    def generator(self, distribution_name):
        """ Get generator for a distribution type """
        return getattr(self, distribution_name)

    def generate_parameters(self, size: int, param_keys: List[str], randomized_list: List[str],
                            param_default: Dict, param_range: Dict, param_dist: Dict):
        param_array = {}
        for key in param_keys:
            if key in randomized_list:
                param_array[key] = self.generator(param_dist[key])(param_range[key], size=size)
            else:
                param_array[key] = np.full(size, param_default[key])
        return param_array
