import torch
import torch.nn as nn
import torch.nn.functional as F

class SummaryNet(nn.Module):
    def __init__(self, nelec, window_size, nstats=24):
        super(SummaryNet, self).__init__()
        
        self.nelec = nelec
        self.window_size = window_size
        self.nout = window_size*self.nelec
        self.nstats = nstats
        
        # 96x96 -> 96x96
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1, padding_mode='replicate')
        # 96x96 -> 48x48
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 48x48 -> 48x48
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, padding_mode='replicate')
        # 48x48 -> 24x24
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 24x24 -> 24x24
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, padding_mode='replicate')
        # 24x24 -> 12x12
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 12x12 -> 10x10
        self.conv4 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3)
        # 10x10 -> 5x5
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # Fully connected layer taking as input the 8 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=125, out_features=24) # 5*5*5=125
        self.fc2 = nn.Linear(in_features=24+self.nstats, out_features=12)

    def forward(self,x):
        x0 = x[:,self.nout:] # n x 24
        x = x[:,:self.nout] # n x 9216
        x = x.view(-1,1,self.window_size,self.nelec) # (batch size,in_channels,height,length) -1 means not changing size of that dimension
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1,125) # (batch size, in_features)
        x = self.fc(x)
        x = torch.cat((x,x0),dim=1)
        # x = self.fc2(x)
        return x


class SummaryNet3D(nn.Module):
    def __init__(self, nelec, window_size, nstats=31):
        super(SummaryNet3D, self).__init__()
        
        self.nelec = nelec
        self.window_size = window_size
        self.nout = window_size*self.nelec
        self.nstats = nstats
        
        # 96x4x96 -> 96x4x190
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=15, kernel_size=3, padding=1, padding_mode='replicate')
        # 96x4x96 -> 48x2x48
        self.conv3d1x1 = nn.Conv3d(in_channels=15, out_channels=15, kernel_size=1, stride=2)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        # 48x2x48 -> 48x2x48
        self.conv2 = nn.Conv3d(in_channels=15, out_channels=15, kernel_size=3, padding=1, padding_mode='replicate')
        # 48x2x48 -> 24x1x24 => reformatted to 24x24
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        # 24x24 -> 24x24
        self.conv2d1x1 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=3, padding=1, padding_mode='replicate')
        # 24x24 -> 12x12
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 12x12 -> 10x10
        self.conv4 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=3, padding=1, padding_mode='replicate')
        # 10x10 -> 5x5
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # Fully connected layer taking as input the 8 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=250, out_features=50) # 5*5*5=125
        self.fc2 = nn.Linear(in_features=50+self.nstats, out_features=12)

    def forward(self,x):
#         print(x.shape)
        x0 = x[:,(-self.nstats):] # n x 24
        x = x[:,:(-self.nstats)] # n x 36864
        x = x.view(-1,1,96,4,190) # 96 or 128
        residual = x
        #x = x.view(-1,1,self.window_size,4,self.nelec) # (batch size,in_channels,height,width,length) -1 means not changing size of that dimension
        # x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x += residual
        x = self.conv3d1x1(x)
        residual = x
        # x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x += residual
        x = self.conv3d1x1(x)
        x = x.view(-1,15,24,48) #24 or 32
        residual = x
        # x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv3(x))
        x += residual
        x = self.conv2d1x1(x)
        residual = x
#         print(x.shape)
        # x = self.pool4(F.relu(self.conv4(x)))
        x = F.relu(self.conv4(x))
        x += residual
        x = self.conv2d1x1(x)
        # -1x15x6x12
#         print(x.shape)
        x = x.view(-1,250) #250 or 350 # (batch size, in_features)
        x = self.fc(x)
#         print(x.shape)
#         print(x0.shape)
        x = torch.cat((x,x0),dim=1)
        # x = self.fc2(x)
        return x