import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional, Union, Tuple, List
from enum import Enum

class ActivationTypes(Enum):
    ReLU = 1
    TanH = 2
    Sigmoid = 3
    LeakyReLU = 4

activation_mapping = {
    ActivationTypes.ReLU: nn.ReLU,
    ActivationTypes.TanH: nn.Tanh,
    ActivationTypes.Sigmoid: nn.Sigmoid,
    ActivationTypes.LeakyReLU: nn.LeakyReLU,
    None: nn.ReLU
}

class ConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels: int, out_features: int, num_filters: Optional[Union[Tuple, List]]=None,
                 activation: Optional[ActivationTypes] = None) -> None:
        super(ConvolutionalNetwork, self).__init__()

        self.activation = activation_mapping[activation]

        n_channels = [16, 32, 64, 128, 128, 256, 256, 256, 64, 8]
        if num_filters is not None:
            n = len(n_channels)
            if not hasattr(num_filters, '__len__'):
                num_filters = [num_filters] * n
            for i in range(min(len(num_filters), n)):
                n_channels[i] = num_filters[i]

        n = 0 # number of layers
        # 3D convolutional block
        conv1 = []
        # input (samp, chan, T, X, Y) ==> (N, 2, 144, 4, 99)
        conv1.append(nn.Conv3d(in_channels=in_channels, out_channels=n_channels[0], 
                               kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='replicate'))
        # (N, 16, 144, 4, 99), downsample X
        conv1.append(nn.Conv3d(in_channels=n_channels[0], out_channels=n_channels[1], 
                               kernel_size=(3, 3, 3), padding=(1, 0, 0), padding_mode='replicate'))
        # (N, 32, 144, 2, 97), downsample T
        conv1.append(nn.Conv3d(in_channels=n_channels[1], out_channels=n_channels[2], 
                               kernel_size=(3, 2, 3), stride=(2, 1, 1), padding=(0, 0, 0), padding_mode='replicate'))
        # (N, 64, 71, 1, 95), dimension reduction
        conv1_dict = OrderedDict()
        for i, x in enumerate(conv1):
            conv1_dict['conv' + str(n + i)] = x
            conv1_dict['actv' + str(n + i)] = self.activation()
        self.conv3d_block = nn.Sequential(conv1_dict)
        n += len(conv1)

        # 2D convolutional block
        conv2 = []
        #(N, 64, 71, 95), downsample
        conv2.append(nn.Conv2d(in_channels=n_channels[2], out_channels=n_channels[3], 
                               kernel_size=(3, 3), stride=2))
        #(N, 128, 35, 47), conv
        conv2.append(nn.Conv2d(in_channels=n_channels[3], out_channels=n_channels[4], 
                               kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'))
        #(N, 128, 35, 47), downsample
        conv2.append(nn.Conv2d(in_channels=n_channels[4], out_channels=n_channels[5], 
                               kernel_size=(3, 3), stride=2))
        #(N, 256, 17, 23), conv
        conv2.append(nn.Conv2d(in_channels=n_channels[5], out_channels=n_channels[6], 
                               kernel_size=(3, 3), padding=(0, 0), padding_mode='replicate'))
        #(N, 256, 15, 21), downsample
        conv2.append(nn.Conv2d(in_channels=n_channels[6], out_channels=n_channels[7], 
                               kernel_size=(3, 3), stride=2))
        #(N, 256, 7, 10)
        conv2_dict = OrderedDict()
        for i, x in enumerate(conv2):
            conv2_dict['conv' + str(n + i)] = x
            conv2_dict['actv' + str(n + i)] = self.activation()
        self.conv2d_block = nn.Sequential(conv2_dict)
        n += len(conv2)

        # 1x1 convolutional block
        conv3 = []
        #(N, 256, 7, 10)
        conv3.append(nn.Conv2d(in_channels=n_channels[7], out_channels=n_channels[8], kernel_size=(1, 1)))
        #(N, 64, 7, 10)
        conv3.append(nn.Conv2d(in_channels=n_channels[8], out_channels=n_channels[9], kernel_size=(1, 1)))
        #(N, 8, 7, 10)
        conv3_dict = OrderedDict()
        for i, x in enumerate(conv3):
            conv3_dict['conv' + str(n + i)] = x
            conv3_dict['actv' + str(n + i)] = self.activation()
        self.conv1x1_block = nn.Sequential(conv3_dict)
        n += len(conv3)

        linear = []
        linear.append(nn.Linear(7 * 10 * n_channels[9], 128))
        linear.append(nn.Linear(128, 32))
        linear.append(nn.Linear(32, out_features))
        linear_dict = OrderedDict()
        for i, x in enumerate(linear[:-1]):
            linear_dict['linear' + str(n + i)] = x
            linear_dict['actv' + str(n + i)] = self.activation()
        n += len(linear)
        linear_dict['linear' + str(n - 1)] = linear[-1]
        self.linear_block = nn.Sequential(linear_dict)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Input Shape: {}".format(x.shape))
        x = self.conv3d_block(x)
        # print("First Shape: {}".format(x.shape))
        
        x = x.squeeze(dim=3) # dimension reduction
        # print("Squeezed Shape: {}".format(x.shape))
        x = self.conv2d_block(x)
        # print("Second Shape: {}".format(x.shape))
        
        x = self.conv1x1_block(x)
        # print("Third Shape: {}".format(x.shape))
        
        x = x.view(x.shape[0], -1) # Flattening
        x = self.linear_block(x)
        # print("Final Shape: {}".format(x.shape))
        return x
