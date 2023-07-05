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

class CircularPad(nn.Module):
    def __init__(self, pad):
        super(CircularPad, self).__init__()
        self.circ_pad = F.pad
        self.pad = pad
        self.mode = 'circular'
        
    def forward(self, x):
        x = self.circ_pad(x, pad=self.pad, mode=self.mode)
        return x

class AutoUNet_half(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        # input: 100x100x1 with initial circular padding
        
        self.e11 = nn.Conv2d(2, 64, kernel_size=3, padding=0, groups=2) # output: 98x98x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=0, groups=2) # output: 96x96x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 48x48x64

        # input: 48x48x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding='same', groups=2) # output: 48x48x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=2) # output: 48x48x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 24x24x128

        # input: 24x24x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding='same', groups=2) # output: 24x24x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=2) # output: 24x24x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 12x12x256

        # input: 12x12x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding='same', groups=2) # output: 12x12x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding='same', groups=2) # output: 12x12x512

        # Decoder
        
        # input: 12x12x256
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # output: 24x24x512
        self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding='same')
        self.d12 = nn.Conv2d(256, 256, kernel_size=3, padding='same')

        # input: 24x24x256
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # output: 48x48x128
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding='same')

        # input: 48x48x128
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # output: 96x96x128
        self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=2) # output: 98x98x128
        self.d32 = nn.Conv2d(64, 64, kernel_size=3, padding=2) # output: 100x100x128

        # Output layer
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        
        x_pad = F.pad(x, (0, 1, 0, 1), mode = 'circular')

        # Encoder
        xe11 = F.relu(self.e11(x_pad))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        
        # Decoder
        xu1 = self.upconv1(xe42)
        xu11 = torch.cat([xu1, xe32], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe22], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe12], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        # Output layer
        out = self.outconv(xd32)[:,:,:-1,:-1]
        
        # Normalization
        out_norm = F.sigmoid(out)

        return out_norm