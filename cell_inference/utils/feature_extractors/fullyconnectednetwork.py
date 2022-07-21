import torch
import torch.nn as nn
from typing import Optional, Union, List
from collections import OrderedDict
from enum import Enum


class ActivationTypes(Enum):
    ReLU = 1
    TanH = 2
    Sigmoid = 3
    LeakyReLU = 4


class FullyConnectedNetwork(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 hidden_layers: int = 2, hidden_layer_size: Union[int,List[int]] = None,
                 activation: Optional[ActivationTypes] = None) -> None:

        self.__activation_mapping = {ActivationTypes.ReLU: nn.ReLU,
                                     ActivationTypes.TanH: nn.Tanh,
                                     ActivationTypes.Sigmoid: nn.Sigmoid,
                                     ActivationTypes.LeakyReLU: nn.LeakyReLU,
                                     None: nn.ReLU}

        super(FullyConnectedNetwork, self).__init__()
        if hidden_layer_size is None:
            hidden_layer_size = [64, 124, 64, 32, 16]
        if hasattr(hidden_layer_size, '__len__'):
            hidden_layers = len(hidden_layer_size)
        else:
            hidden_layer_size = [hidden_layer_size] * hidden_layers
        hidden_layer_size = [in_features] + list(hidden_layer_size)
        self.activation = self.__activation_mapping[activation]

        layers = OrderedDict()
        for i in range(hidden_layers):
            layers[str(i*2)] = nn.Linear(in_features=hidden_layer_size[i], out_features=hidden_layer_size[i+1])
            layers[str(i*2+1)] = self.activation()
        self.hidden_layers = nn.Sequential(layers)

        self.output_layers = nn.Linear(in_features=hidden_layer_size[-1], out_features=out_features)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        x = self.output_layers(x)
        return x
