import torch
import torch.nn as nn
from typing import Optional
from enum import Enum


class ActivationTypes(Enum):
    ReLU = 1
    TanH = 2
    Sigmoid = 3
    LeakyReLU = 4


# TODO Need to finish the network to hold inputs
class ConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels: int,
                 num_labels: int,
                 activation: Optional[ActivationTypes] = None) -> None:
        super(ConvolutionalNetwork, self).__init__()

        self.__activation_mapping = {ActivationTypes.ReLU: nn.ReLU(),
                                     ActivationTypes.TanH: nn.Tanh(),
                                     ActivationTypes.Sigmoid: nn.Sigmoid(),
                                     ActivationTypes.LeakyReLU: nn.LeakyReLU(),
                                     None: nn.ReLU()}

        self.activation = self.__activation_mapping[activation]

        self.input_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=64,
                                    kernel_size=(5, 7),
                                    stride=(1, 1),
                                    padding=(0, 0),
                                    dilation=(1, 1),
                                    padding_mode='replicate')

        self.second_conv = nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=(3, 5),
                                     stride=(1, 2),
                                     padding=(0, 0),
                                     dilation=(1, 1),
                                     padding_mode='replicate')

        self.squaring_conv_block = nn.Sequential(
            nn.Conv2d(64, 64, (1, 5), (1, 1)),
            self.activation,
            nn.Conv2d(64, 64, (1, 5), (1, 1)),
            self.activation,
            nn.Conv2d(64, 64, (1, 5), (1, 1)),
            self.activation,
            nn.Conv2d(64, 64, (1, 3), (1, 1)),
            self.activation,
            nn.Conv2d(64, 64, (1, 3), (1, 1)),
            self.activation
        )
        # Produces (batch, 64, 378, 378)

        self.upchannel_conv_block = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), (1, 1)),
            self.activation,
            nn.Conv2d(128, 256, (3, 3), (1, 1)),
            self.activation,
            nn.Conv2d(256, 512, (3, 3), (1, 1)),
            self.activation
        )
        # Produces (batch, 512, 372, 372)

        self.dilation_conv_block = nn.Sequential(
            nn.Conv2d(512, 512, (5, 5), (1, 1), dilation=(9, 9)),
            self.activation,
            nn.Conv2d(512, 512, (5, 5), (1, 1), dilation=(9, 9)),
            self.activation,
            nn.Conv2d(512, 512, (5, 5), (1, 1), dilation=(9, 9)),
            self.activation
        )
        # Prduces (batch, 512, 270, 270)

        self.stride_conv_block = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), (2, 2)),
            self.activation,
            nn.Conv2d(512, 512, (3, 3), (2, 2)),
            self.activation,
            nn.Conv2d(512, 512, (3, 3), (2, 2)),
            self.activation
        )
        # Produces (batch, 512, 32, 32)

        self.downchannel_conv_block = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1),
            self.activation,
            nn.Conv2d(256, 128, 3, 1),
            self.activation,
            nn.Conv2d(128, 1, 3, 1),
            self.activation
        )
        # Produces (batch, 1, 26, 26)

        self.fc1 = nn.Linear(676, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("Input Shape: {}".format(x.shape))
        x = self.activation(self.input_conv(x))
        x = self.activation(self.second_conv(x))
        print("First Shape: {}".format(x.shape))
        x = self.squaring_conv_block(x)
        print("Second Shape: {}".format(x.shape))
        x = self.upchannel_conv_block(x)
        print("Third Shape: {}".format(x.shape))
        x = self.dilation_conv_block(x)
        x = self.stride_conv_block(x)
        x = self.downchannel_conv_block(x)
        print("Fifth Shape: {}".format(x.shape))
        x = x.view((x.shape[0], -1))  # Flattening
        print("Sixth Shape: {}".format(x.shape))
        x = self.fc1(x)
        print("Final Shape: {}".format(x.shape))
        return x

