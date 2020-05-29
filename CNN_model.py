import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.model = nn.Sequential(
            nn.Conv2d(3, 4, 2, stride=1), #(5-2)/1 + 1=4
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, stride=1), #(4-3)/1 +1 = 2
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2))
        self.linear = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Linear(4, 1))
        
        self.c = nn.Parameter(torch.ones(2))

    def get_coef(self):
        return self.c

    def forward(self, x):   
        y = self.model(x)
        return self.linear(y.squeeze())
