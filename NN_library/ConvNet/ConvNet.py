import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', dev)

class SpecialPad(nn.Module):
    def __init__(self, pad_x, pad_y):
        super().__init__()
        self.pad_x = pad_x
        self.pad_y = pad_y
        
    def forward(self, x):
        x = F.pad(x, self.pad_x, mode = 'circular')
        x = F.pad(x, self.pad_y, mode = 'replicate')
        return x

class ConvNet(nn.Module):
    # Simple ConvNet for stiffness and shrinkage prediction
    def __init__(self, w):
        super().__init__()
        
        self.w = w
        self.conv = nn.Sequential(
            SpecialPad((0, 1, 0, 0), (0, 0, 0, 1)),
            nn.Conv2d(3, self.w, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            SpecialPad((1, 1, 0, 0), (0, 0, 1, 1)),
            nn.Conv2d(self.w, 2*self.w, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            SpecialPad((1, 1, 0, 0), (0, 0, 1, 1)),
            nn.Conv2d(2*self.w, 4*self.w, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            SpecialPad((1, 1, 0, 0), (0, 0, 1, 1)),
            nn.Conv2d(4*self.w, 8*self.w, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            SpecialPad((1, 1, 0, 0), (0, 0, 1, 1)),
            nn.Conv2d(8*self.w, 16*self.w, kernel_size=3),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*self.w * 6 * 6, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        x_conv = self.conv(x)
        out = self.linear(x_conv)

        return out