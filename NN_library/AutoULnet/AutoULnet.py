import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

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

class ULNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        # input: 100x100x1 with initial circular padding
        w = 32

        self.e11 = nn.Conv2d(3, w, kernel_size=3, padding=0) # output: 98x98xw
        self.e12 = nn.Conv2d(w, w, kernel_size=3, padding=0) # output: 96x96xw
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 48x48xw

        # input: 48x48xw
        self.e21 = nn.Conv2d(w, 2*w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 48x48x2w
        self.e22 = nn.Conv2d(2*w, 2*w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 48x48x2w
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 24x24x2w

        # input: 24x24x2w
        self.e31 = nn.Conv2d(2*w, 4*w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 24x24x4w
        self.e32 = nn.Conv2d(4*w, 4*w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 24x24x4w
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 12x12x4w

        # input: 12x12x4w
        self.e41 = nn.Conv2d(4*w, 8*w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 12x12x8w
        self.e42 = nn.Conv2d(8*w, 8*w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 12x12x8w

        # Decoder
        
        # input: 12x12x8w
        self.upconv1 = nn.ConvTranspose2d(8*w, 4*w, kernel_size=2, stride=2) # output: 24x24x4w
        self.d11 = nn.Conv2d(8*w, 4*w, kernel_size=3, padding='same')
        self.d12 = nn.Conv2d(4*w, 4*w, kernel_size=3, padding='same')

        # input: 24x24x256
        self.upconv2 = nn.ConvTranspose2d(4*w, 2*w, kernel_size=2, stride=2) # output: 48x48x2w
        self.d21 = nn.Conv2d(4*w, 2*w, kernel_size=3, padding='same')
        self.d22 = nn.Conv2d(2*w, 2*w, kernel_size=3, padding='same')

        # input: 48x48x128
        self.upconv3 = nn.ConvTranspose2d(2*w, w, kernel_size=2, stride=2) # output: 96x96xw
        self.d31 = nn.Conv2d(2*w, w, kernel_size=3, padding=2) # output: 98x98xw
        self.d32 = nn.Conv2d(w, w, kernel_size=3, padding=2) # output: 100x100xw

        # Output layer
        self.outconv = nn.Conv2d(w, 1, kernel_size=1)

        # L-Encoder
        wl = 16

        # input: 100x100x1
        self.l11a = nn.Conv2d(3, 2*wl, kernel_size=3, padding=0) # output: 98x98x2wl
        self.l12a = nn.Conv2d(2*wl, 2*wl, kernel_size=3, padding=0) # output: 96x96x2wl
        self.l11b = nn.Conv2d(1, wl, kernel_size=3, padding=0) # output: 98x98xwl
        self.l12b = nn.Conv2d(wl, wl, kernel_size=3, padding=0) # output: 96x96xwl
        self.pool1l = nn.MaxPool2d(kernel_size=2, stride=2) # output: 48x48xwl

        # input: 48x48x(3*wl+4w)
        self.l21 = nn.Conv2d(3*wl+4*w, 2*wl+2*w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 48x48x(3*wl+2*w)
        self.l22 = nn.Conv2d(2*wl+2*w, wl+w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 48x48x(wl+w)
        self.pool2l = nn.MaxPool2d(kernel_size=2, stride=2) # output: 24x24x(wl+w)

        # input: 24x24x(wl+w+8w)
        self.l31 = nn.Conv2d(wl+9*w, wl+5*w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 24x24x(wl+w+4w)
        self.l32 = nn.Conv2d(wl+5*w, wl+3*w, kernel_size=3, padding='same', padding_mode = 'circular') # output: 24x24x(wl+w+2w)
        self.pool3l = nn.MaxPool2d(kernel_size=2, stride=2) # output: 12x12x(wl+w+2w)

        # FC
        # input: 12x12x(wl+w+2w)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear((wl+3*w)*12*12, 4096)
        self.fc2 = nn.Linear(4096, 2)

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
        damage = torch.sigmoid(self.outconv(xd32))
        
        # L-encoder
        xl11a = F.relu(self.l11a(x_pad))
        xl12a = F.relu(self.l12a(xl11a))
        xl11b = F.relu(self.l11b(damage))
        xl12b = F.relu(self.l12b(xl11b))
        xl12 = torch.cat([xl12a, xl12b], axis=1)
        xlp1 = self.pool1l(xl12)

        xl2 = torch.cat([xlp1, xu22], axis=1)
        xl21 = F.relu(self.l21(xl2))
        xl22 = F.relu(self.l22(xl21))
        xlp2 = self.pool2l(xl22)

        xl3 = torch.cat([xlp2, xu11], axis=1)
        xl31 = F.relu(self.l31(xl3))
        xl32 = F.relu(self.l32(xl31))
        xlp3 = self.pool3l(xl32)

        xf = self.flat(xlp3)
        xf1 = F.relu(self.fc1(xf))
        xf2 = self.fc2(xf1)

        stiff = torch.sigmoid(xf2[:,0])
        shr = -F.relu(xf[:,1])

        return damage[:,:,:-1,:-1], stiff, shr