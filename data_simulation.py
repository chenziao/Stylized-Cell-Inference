import pandas as pd
import numpy as np
import h5py
import json
from scipy import signal
from scipy.interpolate import LinearNDInterpolator
from typing import Union, List, Tuple, Dict
import os
from tqdm import tqdm

from cell_inference.config import paths, params
from cell_inference.cells.simulation import Simulation
from cell_inference.cells.stylizedcell import CellTypes
from cell_inference.utils.feature_extractors.SummaryStats2D import build_lfp_grid, calculate_stats
from cell_inference.utils.random_parameter_generator import Random_Parameter_Generator
from cell_inference.utils.transform.geometry_transformation import pol2cart, cart2pol
from cell_inference.utils.spike_window import first_pk_tr, get_spike_window


class DataSimulator(object):
    def __init__(self, inference_list=None, number_samples: int = 10000, random_seed: int = 12345):
        self.gmax = None
        self.rand_param = None
        self.labels = None
        if inference_list is None:
            inference_list = ['y', 'd', 'theta', 'h', 'phi', 'r_s', 'l_t', 'r_t']
        self.sim = None
        self.geo_standard = pd.read_csv(paths.GEO_STANDARD, index_col='id')
        self.loc_param_list = ['x', 'y', 'z', 'alpha', 'h', 'phi']
        self.geo_param_list = ['r_s', 'l_t', 'r_t', 'r_d', 'r_tu', 'l_d']

        self.loc_param_default = {'x': 0., 'y': 0., 'z': 50.,
                                  'alpha': np.pi / 4, 'h': 1., 'phi': 0.}

        self.loc_param_default['d'], self.loc_param_default['theta'] = cart2pol(self.loc_param_default['x'],
                                                                                self.loc_param_default['z'])

        self.geo_param_default = {'r_s': 8., 'l_t': 600., 'r_t': 1.25, 'r_d': .28, 'r_tu': .28, 'l_d': 200.}

        self.loc_param_range = {'x': (-50, 50), 'y': (-1400, 1400), 'z': (20., 200.),
                                'alpha': (0, np.pi), 'h': (-1., 1.), 'phi': (-np.pi, np.pi),
                                'd': (20., 200.), 'theta': (-np.pi / 3, np.pi / 3)}
        self.geo_param_range = {'r_s': (7., 12.), 'l_t': (20., 800.), 'r_t': (.6, 1.8),
                                'r_d': (.1, .8), 'r_tu': (.1, .8), 'l_d': (100., 300.)}

        self.loc_param_dist = {'x': 'unif', 'y': 'unif', 'z': 'unif',
                               'alpha': 'unif', 'h': 'unif', 'phi': 'unif', 'd': 'unif', 'theta': 'norm'}
        self.geo_param_dist = {'r_s': 'unif', 'l_t': 'unif', 'r_t': 'unif',
                               'r_d': 'logn', 'r_tu': 'logn', 'l_d': 'unif'}

        self.number_samples = number_samples
        self.rand_seed = random_seed

        self.inference_list = inference_list  # can use d, theta instead of x, z to represent location
        self.randomized_list = ['alpha']  # randomized parameters not to inferred
        self.randomized_list += self.inference_list
        # parameters not in the two lists above are fixed at default.

        self.gmax_mapping = h5py.File(paths.GMAX_MAPPING, 'r')

        self.geo_list = [self.geo_param_list[idx] for idx in self.gmax_mapping['settings/geo_index']]
        for i, key in enumerate(self.geo_list):
            geo_range = self.gmax_mapping['settings/geo_range'][i, :].copy()
            geo_range[0] = max(self.geo_param_range[key][0], geo_range[0])
            geo_range[1] = min(self.geo_param_range[key][1], geo_range[1])
            self.geo_param_range[key] = tuple(geo_range)

        self.squared_soma_radius = self.gmax_mapping['mapping'].attrs['squared_soma_radius']

        # Use linear interpolation
        self.gmax_interp = LinearNDInterpolator(self.gmax_mapping['mapping/geometry'][()],
                                                self.gmax_mapping['mapping/gmax'][()])

        self.gmax_mapping.close()

        self.config_dict = {
            'Trial_Parameters': {'number_samples': self.number_samples, 'rand_seed': self.rand_seed,
                                 'inference_list': self.inference_list, 'randomized_list': self.randomized_list},
            'Simulation_Parameters': {'loc_param_list': self.loc_param_list, 'geo_param_list': self.geo_param_list,
                                      'loc_param_default': self.loc_param_default,
                                      'geo_param_default': self.geo_param_default,
                                      'loc_param_range': self.loc_param_range, 'geo_param_range': self.geo_param_range,
                                      'loc_param_dist': self.loc_param_dist, 'geo_param_dist': self.geo_param_dist}
        }

        self.rpg = Random_Parameter_Generator(seed=rand_seed, n_sigma=3)

    def pred_gmax(self, geo_samples: Dict):
        geo = []
        for k in self.geo_list:
            if self.squared_soma_radius and k == 'r_s':
                geo.append(geo_samples[k] ** 2)
            else:
                geo.append(geo_samples[k])
        gm = self.gmax_interp(np.column_stack(geo))
        return gm

    def simulate_params(self, data_path: str = '10000s_y1Loc2Alt_Ori2_Geo3_params', iteration: int = 1):
        loc_param_gen = self.loc_param_list.copy()
        if 'd' in self.randomized_list and 'theta' in self.randomized_list:
            loc_param_gen[loc_param_gen.index('x')] = 'd'
            loc_param_gen[loc_param_gen.index('z')] = 'theta'

        loc_param_samples = self.rpg.generate_parameters(self.number_samples,
                                                     loc_param_gen,
                                                     self.randomized_list,
                                                     self.loc_param_default,
                                                     self.loc_param_range,
                                                     self.loc_param_dist)

        if 'd' in self.randomized_list and 'theta' in self.randomized_list:
            loc_param_samples['x'], loc_param_samples['z'] = pol2cart(loc_param_samples['d'],
                                                                      loc_param_samples['theta'])

        loc_param = np.column_stack([loc_param_samples[key] for key in self.loc_param_list])

        geo_param_samples = self.rpg.generate_parameters(self.number_samples,
                                                     self.geo_param_list,
                                                     self.randomized_list,
                                                     self.geo_param_default,
                                                     self.geo_param_range,
                                                     self.geo_param_dist)

        geo_param = np.column_stack([geo_param_samples[key] for key in self.geo_param_list])

        self.gmax = self.pred_gmax(geo_param_samples)

        samples = {**geo_param_samples, **loc_param_samples}
        self.labels = np.column_stack([samples[key] for key in self.inference_list])
        self.rand_param = np.column_stack([samples[key] for key in self.randomized_list[:-len(self.inference_list)]])

        np.set_printoptions(suppress=True)

        self.sim = Simulation(geometry=self.geo_standard,
                              electrodes=params.ELECTRODE_POSITION,
                              cell_type=CellTypes.ACTIVE,
                              loc_param=loc_param,
                              geo_param=geo_param,
                              spike_threshold=-30,
                              gmax=self.gmax,
                              scale=1.,
                              ncell=self.number_samples)

        self.sim.run_neuron_sim()
        self.verify_and_save(data_path=data_path, iteration=iteration)


    @staticmethod
    def invalid_index(simulation):
        # index of valid spiking cells
        nspk, tspike = simulation.get_spike_number('all')
        invalid = np.nonzero(nspk != 1)[0]
        return invalid, tspike

    def verify_and_save(self, sim: Simulation = None,
                        data_path: str = '1000s_y1Loc2Alt_Ori2_Geo3_params',
                        iteration: int = 1):
        if sim is not None:
            self.sim = sim
        invalid_idx, tspk = self.invalid_index(self.sim)
        tqdm.write("Number of invalid samples: %d out of %d" % (invalid_idx.size, self.number_samples))

        mem_volt = self.sim.v('all')

        lfp = self.sim.get_lfp('all').transpose((0, 2, 1))  # (cells x channels x time) -> (cells x time x channels)

        filt_b, filt_a = signal.butter(params.BUTTERWORTH_ORDER,
                                       params.FILTER_CRITICAL_FREQUENCY,
                                       params.BANDFILTER_TYPE,
                                       fs=params.FILTER_SAMPLING_RATE)

        filtered_lfp = signal.lfilter(filt_b, filt_a, lfp, axis=1)  # filter along time axis

        pk_tr_idx_in_window = 16  # 16*0.025=0.4 ms
        lfp_list = []
        for i in range(self.number_samples):
            #     filtered_lfp[i] /= np.max(np.abs(filtered_lfp[i]))
            fst_idx = first_pk_tr(filtered_lfp[i])
            start, end = get_spike_window(filtered_lfp[i], win_size=params.WINDOW_SIZE, align_at=pk_tr_idx_in_window)
            lfp_list.append(filtered_lfp[i, start:end, :])

        t = self.sim.t()[:params.WINDOW_SIZE]
        windowed_lfp = np.stack(lfp_list, axis=0)  # (samples x time window x channels)

        sim_data_path = 'cell_inference/resources/simulation_data'
        trial_path = os.path.join(sim_data_path, data_path)

        trial_config_path = os.path.join(trial_path, 'config' + str(iteration) + '.json')  # trial configuration
        lfp_path = os.path.join(trial_path, 'lfp' + str(iteration))  # LFP and labels
        stats_path = os.path.join(trial_path, 'summ_stats' + str(iteration))
        mem_volt_path = os.path.join(trial_path, 'mem_volt' + str(iteration))  # membrane voltage and spike times

        if not os.path.exists(sim_data_path):
            os.makedirs(sim_data_path)
            tqdm.write("The new data directory is created!")

        if not os.path.exists(trial_path):
            os.makedirs(trial_path)
            tqdm.write("The new trial directory is created!")

        summ_stats = []
        bad_indices = []
        yshift = []
        for i in range(windowed_lfp.shape[0]):
            try:
                g_lfp, _, y_i = build_lfp_grid(windowed_lfp[i, :, :], params.ELECTRODE_POSITION[:, :2], y_window_size=960.0)
            except ValueError:
                bad_indices.append(i)
                continue
            summ_stats.append(calculate_stats(g_lfp))
            yshift.append(y_i - self.labels[i, 0])

        summ_stats = np.array(summ_stats)
        yshift = np.array(yshift)
        windowed_lfp = np.delete(windowed_lfp, bad_indices, axis=0)
        self.labels = np.delete(self.labels, bad_indices, axis=0)

        tqdm.write(str(summ_stats.shape))
        tqdm.write(str(self.labels.shape))
        np.savez(lfp_path, t=t, x=windowed_lfp, y=self.labels, ys=yshift, rand_param=self.rand_param, gmax=self.gmax)
        np.savez(stats_path, t=t, x=summ_stats, y=self.labels, ys=yshift, rand_param=self.rand_param, gmax=self.gmax)
        np.savez(mem_volt_path, v=mem_volt, spk=tspk)
        with open(trial_config_path, 'w') as fout:
            json.dump(self.config_dict, fout, indent=2)
