import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', dev)

class CircularPad(nn.Module):
    def __init__(self, pad):
        super(CircularPad, self).__init__()
        self.circ_pad = F.pad
        self.pad = pad
        self.mode = 'circular'
        
    def forward(self, x):
        x = self.circ_pad(x, pad=self.pad, mode=self.mode)
        return x

class ConvNet(nn.Module):
    # Simple ConvNet for stiffness and shrinkage prediction
    def __init__(self, w):
        super().__init__()
        
        self.w = w
        self.conv = nn.Sequential(
            nn.Conv2d(3, self.w, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.w, 2*self.w, kernel_size=3, padding='same', padding_mode = 'circular'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2*self.w, 4*self.w, kernel_size=3, padding='same', padding_mode = 'circular'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4*self.w, 8*self.w, kernel_size=3, padding='same', padding_mode = 'circular'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8*self.w, 16*self.w, kernel_size=3, padding='same', padding_mode = 'circular'),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*self.w * 6 * 6, 2),
            nn.ReLU()
        )

    def forward(self, x):
        
        x_pad = F.pad(x, (0, 1, 0, 1), mode = 'circular')
        x_conv = self.conv(x_pad)
        out = self.linear(x_conv)

        return out