import numpy as np
from typing import Optional, List, Tuple, Dict
from cell_inference.utils.transform.distribution_transformation import range2norm, range2logn
from cell_inference.utils.transform.geometry_transformation import pol2cart

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


def generate_parameters_from_config(config: Dict):
    """ Generate parameters from configuration dictionary """
    tr_p = config['Trial_Parameters']
    sim_p = config['Simulation_Parameters']
    n_sigma = sim_p.get('n_sigma')
    if n_sigma is None: n_sigma = 3.0
    batch_id = sim_p.get('batch_id')
    if batch_id is None: batch_id = 0
    rpg = Random_Parameter_Generator(seed=tr_p['rand_seed'] + batch_id, n_sigma=n_sigma)
    
    # Location paramters
    loc_param_gen = sim_p['loc_param_list'].copy()
    polar_loc = 'd' in tr_p['randomized_list'] and 'theta' in tr_p['randomized_list']
    if polar_loc:
        loc_param_gen[loc_param_gen.index('x')] = 'd'
        loc_param_gen[loc_param_gen.index('z')] = 'theta'

    loc_param_samples = rpg.generate_parameters(
        tr_p['number_samples'], loc_param_gen, tr_p['randomized_list'],
        sim_p['loc_param_default'], sim_p['loc_param_range'], sim_p['loc_param_dist']
    )

    if polar_loc:
        loc_param_samples['x'], loc_param_samples['z'] = pol2cart(loc_param_samples['d'], loc_param_samples['theta'])

    loc_param = np.column_stack([loc_param_samples[key] for key in sim_p['loc_param_list']])

    # reshape into ncell-by-nloc-by-nparam
    loc_param = loc_param.reshape(tr_p['number_cells'], tr_p['number_locs'], -1)

    # Geometery parameters
    geo_param_samples = rpg.generate_parameters(
        tr_p['number_cells'], sim_p['geo_param_list'], tr_p['randomized_list'],
        sim_p['geo_param_default'], sim_p['geo_param_range'], sim_p['geo_param_dist']
    )

    geo_param = np.column_stack([geo_param_samples[key] for key in sim_p['geo_param_list']])

    # repeat to match number_samples
    for key, value in geo_param_samples.items():
        geo_param_samples[key] = np.repeat(value, tr_p['number_locs'])
    
    # Gather parameters as labels
    samples = {**geo_param_samples, **loc_param_samples}
    labels = np.column_stack([ samples[key] for key in tr_p['inference_list'] ])
    rand_param = np.column_stack([ samples[key] for key in tr_p['randomized_list'][:-len(tr_p['inference_list'])] ])
    return labels, rand_param, loc_param, geo_param

def generate_predicted_parameters_from_config(config: Dict, pred_dict: Dict, number_locs: int = 1):
    """ Generate parameters from configuration and prediction dictionary """
    tr_p = config['Trial_Parameters']
    sim_p = config['Simulation_Parameters']
    n_sigma = sim_p.get('n_sigma')
    if n_sigma is None: n_sigma = 3.0
    rpg = Random_Parameter_Generator(seed=tr_p['rand_seed'], n_sigma=n_sigma)

    # Clip predicted parameters
    pred_param = {}
    for key, p_range in {**sim_p['loc_param_range'], **sim_p['geo_param_range']}.items():
        if key in pred_dict:
            pred_param[key] = np.clip(pred_dict[key], p_range[0], p_range[1])
            number_cells = pred_param[key].size

    # Location paramters
    loc_param_gen = sim_p['loc_param_list'].copy()
    if 'd' in tr_p['randomized_list'] and 'theta' in tr_p['randomized_list']:
        loc_param_gen[loc_param_gen.index('x')] = 'd'
        loc_param_gen[loc_param_gen.index('z')] = 'theta'

    # predicted
    loc_param_samples = {}
    for key, value in pred_param.items():
        if key in loc_param_gen:
            loc_param_samples[key] = np.repeat(value, number_locs)
            loc_param_gen.remove(key)
    # randomized
    loc_param_samples.update(rpg.generate_parameters(
        number_cells * number_locs, loc_param_gen, tr_p['randomized_list'],
        sim_p['loc_param_default'], sim_p['loc_param_range'], sim_p['loc_param_dist']
    ))

    if 'd' in loc_param_samples and 'theta' in loc_param_samples:
        loc_param_samples['x'], loc_param_samples['z'] = pol2cart(loc_param_samples['d'], loc_param_samples['theta'])

    loc_param = np.column_stack([loc_param_samples[key] for key in sim_p['loc_param_list']])

    # reshape into ncell-by-nloc-by-nparam
    loc_param = loc_param.reshape(number_cells, number_locs, -1)

    # Geometery parameters
    geo_param_gen = sim_p['geo_param_list'].copy()
    geo_param_samples = {}
    for key, value in pred_param.items():
        if key in geo_param_gen:
            geo_param_samples[key] = value
            geo_param_gen.remove(key)
    geo_param_samples.update(rpg.generate_parameters(
        number_cells, geo_param_gen, tr_p['randomized_list'],
        sim_p['geo_param_default'], sim_p['geo_param_range'], sim_p['geo_param_dist']
    ))

    geo_param = np.column_stack([geo_param_samples[key] for key in sim_p['geo_param_list']])

    # repeat to match number_samples
    for key, value in geo_param_samples.items():
        geo_param_samples[key] = np.repeat(value, number_locs)
    
    # Gather parameters as labels
    samples = {**geo_param_samples, **loc_param_samples}
    labels = np.column_stack([ samples[key] for key in tr_p['inference_list'] ])[::number_locs, :]
    rand_param = np.column_stack([ samples[key] for key in tr_p['randomized_list'][:-len(tr_p['inference_list'])] ])
    rand_param = rand_param.reshape(number_cells, number_locs, -1)
    return labels, rand_param, loc_param, geo_param
