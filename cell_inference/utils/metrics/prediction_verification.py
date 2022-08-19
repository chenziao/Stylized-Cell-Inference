import pandas as pd
import numpy as np
import h5py
from scipy import signal
from scipy.interpolate import LinearNDInterpolator
from typing import Dict
import os
from tqdm import tqdm

from cell_inference.config import paths, params
from cell_inference.cells.simulation import Simulation
from cell_inference.cells.stylizedcell import CellTypes
from cell_inference.utils.transform.geometry_transformation import pol2cart, cart2pol
from cell_inference.utils.spike_window import get_spike_window


class InVivoParamSimulator(object):
    def __init__(self, data: pd.DataFrame, inference_list: str = None, data_path: str = None):
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

        self.gmax_mapping = h5py.File(paths.GMAX_MAPPING, 'r')
        self.geo_list = [self.geo_param_list[idx] for idx in self.gmax_mapping['settings/geo_index']]
        self.squared_soma_radius = self.gmax_mapping['mapping'].attrs['squared_soma_radius']
        # Use linear interpolation
        self.gmax_interp = LinearNDInterpolator(self.gmax_mapping['mapping/geometry'][()],
                                                self.gmax_mapping['mapping/gmax'][()])
        self.gmax_mapping.close()

        x, z = pol2cart(data['d'].to_numpy(), data['theta'].to_numpy())

        self.data = {'x': x, 'y': data['y'].to_numpy(), 'z': z,
                     'alpha': np.tile(self.loc_param_default['alpha'], x.shape[0]),
                     'h': data['h'].to_numpy(), 'phi': data['phi'].to_numpy(), 'r_s': data['r_s'].to_numpy(),
                     'l_t': data['l_t'].to_numpy(), 'r_t': data['r_t'].to_numpy(),
                     'r_d': np.tile(self.geo_param_default['r_d'], x.shape[0]),
                     'r_tu': np.tile(self.geo_param_default['r_tu'], x.shape[0]),
                     'l_d': np.tile(self.geo_param_default['l_d'], x.shape[0])}

        self.loc_param_samples = {k: self.data[k] for k in ('x', 'y', 'z', 'alpha', 'h', 'phi') if
                                  k in self.data}

        self.geo_param_samples = {k: self.data[k] for k in ('r_s', 'l_t', 'r_t', 'r_d', 'r_tu', 'l_d') if
                                  k in self.data}

        self.gmax = self.pred_gmax(self.geo_param_samples)

        self.loc_param_samples = pd.DataFrame.from_dict(self.loc_param_samples).to_numpy()
        self.geo_param_samples = pd.DataFrame.from_dict(self.geo_param_samples).to_numpy()

        self.inference_list = inference_list  # can use d, theta instead of x, z to represent location
        self.randomized_list = ['alpha']  # randomized parameters not to inferred
        self.randomized_list += self.inference_list
        # parameters not in the two lists above are fixed at default.

        np.set_printoptions(suppress=True)

        self.sim = Simulation(geometry=self.geo_standard,
                              electrodes=params.ELECTRODE_POSITION,
                              cell_type=CellTypes.ACTIVE,
                              loc_param=self.loc_param_samples,
                              geo_param=self.geo_param_samples,
                              spike_threshold=params.SPIKE_THRESHOLD,
                              gmax=self.gmax,
                              scale=1.,
                              ncell=x.shape[0])

        self.sim.run_neuron_sim()
        # self.verify_and_save(data_path=data_path, save=False)

    def pred_gmax(self, geo_samples: Dict):
        geo = []
        for k in self.geo_list:
            if self.squared_soma_radius and k == 'r_s':
                geo.append(geo_samples[k] ** 2)
            else:
                geo.append(geo_samples[k])
        gm = self.gmax_interp(np.column_stack(geo))
        return gm

    @staticmethod
    def invalid_index(simulation):
        # index of valid spiking cells
        nspk, tspike = simulation.get_spike_number('all')
        print(nspk)
        invalid = np.nonzero(nspk != 1)[0]
        return invalid, tspike

    def verify_and_save(self, sim: Simulation = None,
                        data_path: str = '1000s_y1Loc2Alt_Ori2_Geo3_params',
                        save: bool = True):
        if sim is not None:
            self.sim = sim
        invalid_idx, tspk = self.invalid_index(self.sim)
        tqdm.write("Number of invalid samples: %d out of %d" % (invalid_idx.size, self.sim.ncell))

        mem_volt = self.sim.v('all')

        lfp = self.sim.get_lfp('all').transpose((0, 2, 1))  # (cells x channels x time) -> (cells x time x channels)

        filt_b, filt_a = signal.butter(params.BUTTERWORTH_ORDER,
                                       params.FILTER_CRITICAL_FREQUENCY,
                                       params.BANDFILTER_TYPE,
                                       fs=params.FILTER_SAMPLING_RATE)

        filtered_lfp = signal.lfilter(filt_b, filt_a, lfp, axis=1)  # filter along time axis

        pk_tr_idx_in_window = 16  # 16*0.025=0.4 ms
        lfp_list = []

        for i in range(self.sim.ncell):
            start, end = get_spike_window(filtered_lfp[i], win_size=params.WINDOW_SIZE, align_at=pk_tr_idx_in_window)
            lfp_list.append(filtered_lfp[i, start:end, :])

        t = self.sim.t()[:params.WINDOW_SIZE]
        windowed_lfp = np.stack(lfp_list, axis=0)  # (samples x time window x channels)

        sim_data_path = 'cell_inference/resources/simulation_data'
        trial_path = os.path.join(sim_data_path, data_path)

        lfp_path = os.path.join(trial_path, 'invivo_lfp')  # LFP and labels
        mem_volt_path = os.path.join(trial_path, 'invivo_mem_volt')  # membrane voltage and spike times

        if not os.path.exists(sim_data_path):
            os.makedirs(sim_data_path)
            tqdm.write("The new data directory is created!")

        if not os.path.exists(trial_path):
            os.makedirs(trial_path)
            tqdm.write("The new trial directory is created!")
        if save:
            np.savez(lfp_path, t=t, x=windowed_lfp, y=self.labels, rand_param=self.rand_param, gmax=self.gmax)
            np.savez(mem_volt_path, v=mem_volt, spk=tspk)

        return windowed_lfp, t
