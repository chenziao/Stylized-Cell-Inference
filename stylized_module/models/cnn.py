import torch
import torch.nn as nn
import torch.nn.functional as F

class SummaryNet(nn.Module):
    def __init__(self, nelec, window_size):
        super(SummaryNet, self).__init__()
        
        self.nelec = nelec
        self.window_size = window_size
        self.nout = window_size*self.nelec
        
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
        self.fc = nn.Linear(in_features=125, out_features=12) # 5*5*5=125

    def forward(self,x):
        x0 = x[:,self.nout:]
        x = x[:,:self.nout]
        x = x.view(-1,1,self.window_size,self.nelec) # (batch size,in_channels,height,length) -1 means not changing size of that dimension
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1,125) # (batch size, in_features)
        x = F.relu(self.fc(x))
        x = torch.cat((x,x0),dim=1)
        return x