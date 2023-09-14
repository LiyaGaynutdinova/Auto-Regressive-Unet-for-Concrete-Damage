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

class RecUnet(nn.Module):
    def __init__(self):
        #U-net with the LSTM unit at the bottleneck
        super().__init__()
        
        # Encoder
        # input: 100x100x1 with initial circular padding
        self.w = 8

        self.e11 = nn.Conv2d(3, self.w, kernel_size=3, padding=0) # output: 98x98xself.w
        self.e12 = nn.Conv2d(self.w, self.w, kernel_size=3, padding=0) # output: 96x96xw
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 48x48xw

        # input: 48x48xw
        self.e21 = nn.Conv2d(self.w, 2*self.w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 48x48x2w
        self.e22 = nn.Conv2d(2*self.w, 2*self.w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 48x48x2w
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 24x24x2w

        # input: 24x24x2w
        self.e31 = nn.Conv2d(2*self.w, 4*self.w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 24x24x4w
        self.e32 = nn.Conv2d(4*self.w, 4*self.w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 24x24x4w
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 12x12x4w

        # input: 12x12x4w
        self.e41 = nn.Conv2d(4*self.w, 8*self.w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 12x12x8w
        self.e42 = nn.Conv2d(8*self.w, 8*self.w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 12x12x8w
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 6x6x8w

        # input: 6x6x8w
        self.e51 = nn.Conv2d(8*self.w, 8*self.w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 6x6x8w
        self.e52 = nn.Conv2d(8*self.w, 8*self.w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 6x6x8w

        # GRU
        self.gru = nn.GRUCell(input_size=6*6*8*self.w, hidden_size=6*6*8*self.w)

        # FCNN
        self.fc1 = nn.Linear(6*6*8*self.w, 512)
        self.fc2 = nn.Linear(512, 2)

        # Decoder
        # input: 6x6x8w
        self.upconv0 = nn.ConvTranspose2d(8*self.w, 8*self.w, kernel_size=2, stride=2) # output: 12x12x4w
        self.d01 = nn.Conv2d(16*self.w, 8*self.w, kernel_size=3, padding='same')
        self.d02 = nn.Conv2d(8*self.w, 8*self.w, kernel_size=3, padding='same')
        
        # input: 12x12x8w
        self.upconv1 = nn.ConvTranspose2d(8*self.w, 4*self.w, kernel_size=2, stride=2) # output: 24x24x4w
        self.d11 = nn.Conv2d(8*self.w, 4*self.w, kernel_size=3, padding='same')
        self.d12 = nn.Conv2d(4*self.w, 4*self.w, kernel_size=3, padding='same')

        # input: 24x24x4w
        self.upconv2 = nn.ConvTranspose2d(4*self.w, 2*self.w, kernel_size=2, stride=2) # output: 48x48x2w
        self.d21 = nn.Conv2d(4*self.w, 2*self.w, kernel_size=3, padding='same')
        self.d22 = nn.Conv2d(2*self.w, 2*self.w, kernel_size=3, padding='same')

        # input: 48x48x2w
        self.upconv3 = nn.ConvTranspose2d(2*self.w, self.w, kernel_size=2, stride=2) # output: 96x96xw
        self.d31 = nn.Conv2d(2*self.w, self.w, kernel_size=3, padding=2) # output: 98x98xw
        self.d32 = nn.Conv2d(self.w, self.w, kernel_size=3, padding=2) # output: 100x100xw

        # Output layer
        self.outconv = nn.Conv2d(self.w, 1, kernel_size=1)

    def forward(self, x, h_0):
        
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
        xp4 = self.pool3(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        # GRU cell
        x_f = xe52.view(-1, 6*6*8*self.w)
        h_1 = self.gru(x_f, h_0)

        # FCNN
        x_fc1 = F.relu(self.fc1(h_1))
        out_2 = torch.sigmoid(self.fc2(x_fc1))

        # Decoder
        x_h = h_1.view(-1, 8*self.w, 6, 6)

        xu0 = self.upconv0(x_h)
        xu01 = torch.cat([xu0, xe42], dim=1)
        xd01 = F.relu(self.d01(xu01))
        xd02 = F.relu(self.d02(xd01))

        xu1 = self.upconv1(xd02)
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
        out_1 = torch.sigmoid(self.outconv(xd32)[:,:,:-1,:-1])
        
        return out_1, out_2, h_1