import torch.nn as nn
from typing import Optional
from collections import OrderedDict
from enum import Enum


class ActivationTypes(Enum):
    ReLU = 1
    TanH = 2
    Sigmoid = 3
    LeakyReLU = 4


class FullyConnectedNetwork(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 hidden_layers: int = 2, hidden_layer_size: int = 50,
                 activation: Optional[ActivationTypes] = None) -> None:

        self.__activation_mapping = {ActivationTypes.ReLU: nn.ReLU(),
                                     ActivationTypes.TanH: nn.Tanh(),
                                     ActivationTypes.Sigmoid: nn.Sigmoid(),
                                     ActivationTypes.LeakyReLU: nn.LeakyReLU(),
                                     None: nn.ReLU()}

        super(FullyConnectedNetwork, self).__init__()
        self.activation = self.__activation_mapping[activation]
        self.input_layer = nn.Linear(in_features=in_features, out_features=hidden_layer_size)
        layers = OrderedDict()
        for i in range(hidden_layers * 2):
            layers[str(i)] = nn.Linear(hidden_layer_size, hidden_layer_size) if i % 2 == 0 else self.activation

        self.hidden_layers = nn.Sequential(layers)

        self.output_layers = nn.Linear(in_features=50, out_features=out_features)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.hidden_layers(x)
        x = self.output_layers(x)
        return x
