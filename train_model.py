import numpy as np
import os
import glob
from typing import Optional
import json

from cell_inference.cells.stylizedcell import CellTypes
from cell_inference.config import params

from cell_inference.utils.feature_extractors.SummaryStats2D import calculate_stats, build_lfp_grid

from cell_inference.utils.transform.geometry_transformation import hphi2unitsphere
from cell_inference.utils.feature_extractors.helperfunctions import train_regression, build_dataloader_from_numpy
from cell_inference.utils.feature_extractors.fullyconnectednetwork import FullyConnectedNetwork
import torch


class Trainer(object):

    data: Optional[np.ndarray]
    labels: Optional[np.ndarray]
    cell_type: CellTypes

    def __init__(self, trial_path: str = '5000s_Loc2Alt_Ori2_Geo3_params'):

        self.data = None
        self.labels = None
        self.cell_type = CellTypes.ACTIVE

        self.trial_path = os.path.join('cell_inference/resources/simulation_data', trial_path)

        config_paths = glob.glob(os.path.join(self.trial_path, 'config*.json'))

        summ_stat_paths = glob.glob(os.path.join(self.trial_path, 'summ_stats*.npz'))

        # Using Summary Stats as the Input Data
        for ssp in summ_stat_paths:
            with np.load(ssp) as data:
                if self.data is None:
                    self.data = data['x']
                    self.labels = data['y']
                else:
                    self.data = np.concatenate((self.data, data['x']), axis=0)
                    self.labels = np.concatenate((self.labels, data['y']), axis=0)

        with open(config_paths[0], 'r') as f:
            self.config = json.load(f)

        self.inference_list = self.config['Trial_Parameters']['inference_list']
        self.ranges = self.config['Simulation_Parameters']['loc_param_range']
        self.ranges.update(self.config['Simulation_Parameters']['geo_param_range'])

    def convert_hphi_to_dv(self, labels: Optional[np.ndarray] = None) -> np.ndarray:
        if labels is not None:
            self.labels = labels
        # convert_hphi_to_dv = False
        dv = hphi2unitsphere(self.labels)
        dvx, dvy, dvz = tuple(np.hsplit(dv, 3))
        self.labels = np.concatenate((dvx, dvy, dvz), axis=1)
        return self.labels

    def normalize_labels(self, labels: Optional[np.ndarray] = None) -> np.ndarray:
        if labels is not None:
            self.labels = labels
        feature_range = (-1, 1)

        # normalize_labels = True

        # if normalize_labels:
        for i in range(self.labels.shape[1]):
            label = self.labels[:, i]
            label_name = self.inference_list[i]
            min_max_range = self.ranges[label_name]
            x_std = (label - min_max_range[0]) / (min_max_range[1] - min_max_range[0])
            x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
            self.labels[:, i] = x_scaled
        return self.labels

    def build_data_and_fit(self, data: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None):
        if data is not None:
            self.data = data
        if labels is not None:
            self.labels = labels
        train_loader, test_loader = build_dataloader_from_numpy(input_arr=self.data,
                                                                labels_arr=self.labels,
                                                                batch_size=2)

        model = FullyConnectedNetwork(in_features=40, out_features=8)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # torch.device("cpu")
        model.to(device)

        train_regression(model, train_loader, test_loader, 35, learning_rate=0.001, decay_rate=0.99, device=device)
        model.eval()
        torch.save(model.state_dict(), os.path.join(self.trial_path, 'model.pth'))
        return model
