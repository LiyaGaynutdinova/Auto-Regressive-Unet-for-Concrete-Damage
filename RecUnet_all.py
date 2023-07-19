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

class RecUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        # input: 100x100x1 with initial circular padding

        self.e11 = nn.Conv2d(2, 32, kernel_size=3, padding=0) # output: 98x98x32
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=0) # output: 96x96x16

        self.a11 = nn.Conv2d(3, 3*32, kernel_size=3, padding=0, groups=3) # output: 98x98x96
        self.a12 = nn.Conv2d(3*32, 3*32, kernel_size=3, padding=0, groups=3) # output: 96x96x96
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 48x48

        # input: 48x48
        self.e21 = nn.Conv2d(32, 64, kernel_size=3, padding='same', padding_mode = 'circular') # output: 48x48x64
        self.e22 = nn.Conv2d(64, 64, kernel_size=3, padding='same', padding_mode = 'circular') # output: 48x48x64
        
        self.a21 = nn.Conv2d(3*32, 3*64, kernel_size=3, padding='same', padding_mode = 'circular', groups=3) # output: 48x48x192
        self.a22 = nn.Conv2d(3*64, 3*64, kernel_size=3, padding='same', padding_mode = 'circular', groups=3) # output: 48x48x192
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 24x24

        # input: 24x24
        self.e31 = nn.Conv2d(64, 128, kernel_size=3, padding='same', padding_mode = 'circular') # output: 24x24x128
        self.e32 = nn.Conv2d(128, 128, kernel_size=3, padding='same', padding_mode = 'circular') # output: 24x24x128
        
        self.a31 = nn.Conv2d(3*64, 3*128, kernel_size=3, padding='same', padding_mode = 'circular', groups=3) # output: 24x24x384
        self.a32 = nn.Conv2d(3*128, 3*128, kernel_size=3, padding='same', padding_mode = 'circular', groups=3) # output: 24x24x384
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 12x12

        # input: 12x12x256
        self.e41 = nn.Conv2d(128, 256, kernel_size=3, padding='same', padding_mode = 'circular') # output: 12x12x256
        self.e42 = nn.Conv2d(256, 256, kernel_size=3, padding='same', padding_mode = 'circular') # output: 12x12x256
        
        self.a41 = nn.Conv2d(3*128, 3*256, kernel_size=3, padding='same', padding_mode = 'circular', groups=3) # output: 12x12x868
        self.a42 = nn.Conv2d(3*256, 3*256, kernel_size=3, padding='same', padding_mode = 'circular', groups=3) # output: 12x12x868
        
        # Decoder
        
        # input: 12x12x868
        self.upconv1 = nn.ConvTranspose2d(4*256, 4*128, kernel_size=2, stride=2) # output: 24x24x384
        self.d11 = nn.Conv2d(4*256, 4*128, kernel_size=3, padding='same')
        self.d12 = nn.Conv2d(4*128, 4*128, kernel_size=3, padding='same')

        # input: 24x24x256
        self.upconv2 = nn.ConvTranspose2d(4*128, 4*64, kernel_size=2, stride=2) # output: 48x48x192
        self.d21 = nn.Conv2d(4*128, 4*64, kernel_size=3, padding='same')
        self.d22 = nn.Conv2d(4*64, 4*64, kernel_size=3, padding='same')

        # input: 48x48x128
        self.upconv3 = nn.ConvTranspose2d(4*64, 4*32, kernel_size=2, stride=2) # output: 96x96x128
        self.d31 = nn.Conv2d(4*64, 4*32, kernel_size=3, padding=2) # output: 98x98x128
        self.d32 = nn.Conv2d(4*32, 4*32, kernel_size=3, padding=2) # output: 100x100x128

        # Output layer
        self.outconv = nn.Conv2d(4*32, 3, kernel_size=1)

    def forward(self, x):
        
        x_pad = F.pad(x, (0, 1, 0, 1), mode = 'circular')

        # Encoder
        xe11 = F.relu(self.e11(x_pad[:,:2]))
        xe12 = F.relu(self.e12(xe11))
        xa11 = F.relu(self.a11(x_pad[:,2:]))
        xa12 = F.relu(self.a12(xa11))
        xe1 = torch.cat([xe12, xa12], dim=1)
        xp1 = self.pool1(xe1)

        xe21 = F.relu(self.e21(xp1[:,:32]))
        xe22 = F.relu(self.e22(xe21))
        xa21 = F.relu(self.a21(xp1[:,32:]))
        xa22 = F.relu(self.a22(xa21))
        xe2 = torch.cat([xe22, xa22], dim=1)
        xp2 = self.pool2(xe2)

        xe31 = F.relu(self.e31(xp2[:,:64]))
        xe32 = F.relu(self.e32(xe31))
        xa31 = F.relu(self.a31(xp2[:,64:]))
        xa32 = F.relu(self.a32(xa31))
        xe3 = torch.cat([xe32, xa32], dim=1)
        xp3 = self.pool3(xe3)

        xe41 = F.relu(self.e41(xp3[:,:128]))
        xe42 = F.relu(self.e42(xe41))
        xa41 = F.relu(self.a41(xp3[:,128:]))
        xa42 = F.relu(self.a42(xa41))
        xe4 = torch.cat([xe42, xa42], dim=1)
        
        # Decoder
        xu1 = self.upconv1(xe4)
        xu11 = torch.cat([xu1, xe3], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe2], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe1], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        # Output layer
        out = self.outconv(xd32)[:,:,:-1,:-1]
        
        # Damage normalization 
        damage_norm = torch.sigmoid(out[:,[0],:,:])
        out_norm = torch.cat([damage_norm, out[:,1:,:,:]], axis=1)

        return out_norm