import os
import glob
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import torch
import csv
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

  
class dataset_uniform(Dataset):
    def __init__(self, type=None):
        if type=='weird':
            self.imgs_path = "reduxed_results/weird_geometry/geometry/"
            self.label_path = "reduxed_results/weird_geometry/damage_fields/"
            N = 11           
        else:
            self.imgs_path = "reduxed_results/uniform/geometry/"
            self.label_path = "reduxed_results/uniform/damage_fields/"
            N = 15000
        self.img_res = 99
        imp_shrinkage_values = pd.read_csv(self.label_path + 'stiffness_0.csv', sep='\t', usecols=['#shr_imposed[-]']).values.tolist()
        imp_shrinkage_matrices = [np.full((self.img_res, self.img_res), value) for value in imp_shrinkage_values]
        imp_shrinkage_matrices_stacked = np.stack(imp_shrinkage_matrices)
        self.imp_shrinkage = torch.tensor(imp_shrinkage_matrices_stacked, dtype=torch.float)
        self.data = []
        for i in range(N):
            if i != 5000:
                sequence = {}
                sequence['geometry'] = self.imgs_path + str(i) + '.npy'
                sequence['damage'] = []
                for j in range(10):
                    input = self.label_path + str(i) + '_' + str((j+1)*11) + '.npy'
                    sequence['damage'].append(input) 
                sequence['obs_shrinkage'] = pd.read_csv(self.label_path + f'stiffness_{i}.csv', sep='\t', usecols=['#shr_observed[-]']).values.tolist()
                sequence['stiffness'] = pd.read_csv(self.label_path + f'stiffness_{i}.csv', sep='\t', usecols=['#stiffness[MPa]']).values.tolist()
                self.data.append(sequence)       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        img_geometry = np.load(sequence['geometry'])[:-1,:-1]
        img_damage = np.stack([np.load(path)[:-1,:-1] for path in sequence['damage']])
        img_null = np.zeros(((99, 99)))
        img_stacked = np.stack([img_geometry, img_null])
        img_stacked = np.concatenate([img_stacked, img_damage])
        tensor = torch.tensor(img_stacked, dtype=torch.float)
        transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ) 
        tensor = transform(tensor)
        k = np.random.rand()
        if (k < 0.25):
            tensor = transforms.functional.rotate(tensor, 90)
        elif (k >= 0.25) and (k < 0.5):
            tensor = transforms.functional.rotate(tensor, 180)
        elif (k >= 0.5) and (k < 0.75):
            tensor = transforms.functional.rotate(tensor, 270)
        roll_x = np.random.randint(self.img_res)
        roll_y = np.random.randint(self.img_res)
        tensor = torch.roll(tensor, roll_x, -2)
        tensor = torch.roll(tensor, roll_y, -1)
        return (tensor[0].view(1,self.img_res,self.img_res), 
                tensor[1:], 
                self.imp_shrinkage, 
                torch.tensor([sequence['obs_shrinkage']], dtype=torch.float).flatten(),
                torch.tensor([sequence['stiffness']], dtype=torch.float).flatten())


class dataset_big(Dataset):
    def __init__(self, type=None):
        self.imgs_path = "C:/Users/Jorge/OneDrive/ModularOptimization/Concrete/level_set_big/"
        N = 100000
        self.img_res = 99
        imp_shrinkage_values = pd.read_csv('reduxed_results/damage_fields/stiffness_0.csv', sep='\t', usecols=['#shr_imposed[-]']).values.tolist()
        imp_shrinkage_matrices = [np.full((self.img_res, self.img_res), value) for value in imp_shrinkage_values]
        imp_shrinkage_matrices_stacked = np.stack(imp_shrinkage_matrices)
        self.imp_shrinkage = torch.tensor(imp_shrinkage_matrices_stacked, dtype=torch.float)
        self.data = []
        for i in range(N):
            sequence = self.imgs_path + str(i) + '.npy'
            self.data.append(sequence)       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        img_geometry = 1.-np.load(sequence)[:-1,:-1]
        tensor = torch.tensor(img_geometry, dtype=torch.float).view(1,99,99)
        return tensor, self.imp_shrinkage
    
def get_loaders(data, batch_size):
    n_train = int(0.8 * data.__len__())
    n_test = (data.__len__() - n_train) // 2
    n_val = data.__len__() - n_train - n_test
    torch.manual_seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(data, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size = batch_size)
    test_loader = DataLoader(test_set, batch_size = batch_size)
    loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
    return loaders

def get_loaders_manual(data, batch_size):
    dataset_size = len(data)
    indices = list(range(dataset_size))
    n_train = int(0.8 * dataset_size)
    n_val = int(0.1* dataset_size)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:(n_train + n_val)]
    test_indices = indices[(n_train + n_val):]
    train_set = torch.utils.data.Subset(data, train_indices)
    val_set = torch.utils.data.Subset(data, val_indices)
    test_set = torch.utils.data.Subset(data, test_indices)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size = batch_size)
    test_loader = DataLoader(test_set, batch_size = batch_size)
    loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
    return loaders


class dataset_nonuniform(Dataset):
    def __init__(self, type=None):
        self.imgs_path = "reduxed_results/non_uniform/geometry/"
        self.label_path = "reduxed_results/non_uniform/damage_fields/"
        N = 15000          
        self.img_res = 99
        self.imp_shrinkage = [np.zeros((99, 99))]
        for i in range(11, 121, 11):
            self.imp_shrinkage.append(np.flipud(np.load(f'reduxed_results/non_uniform/shrinkage_{i}.npy')[:-1,:-1]))
        self.imp_shrinkage = np.stack(self.imp_shrinkage)
        self.imp_shrinkage = torch.tensor(self.imp_shrinkage, dtype=torch.float)
        self.data = []
        for i in range(N):
            sequence = {}
            sequence['geometry'] = self.imgs_path + str(i) + '.npy'
            sequence['damage'] = []
            for j in range(10):
                input = self.label_path + str(i) + '_' + str((j+1)*11) + '.npy'
                sequence['damage'].append(input) 
            sequence['obs_shrinkage'] = pd.read_csv(self.label_path + f'stiffness_{i}.csv', sep='\t', usecols=['#axial_shrinkage[-]']).values.tolist()
            sequence['stiffness'] = pd.read_csv(self.label_path + f'stiffness_{i}.csv', sep='\t', usecols=['#stiffness[MPa]']).values.tolist()
            self.data.append(sequence)       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        img_geometry = np.load(sequence['geometry'])
        img_damage = np.stack([np.load(path) for path in sequence['damage']])
        img_null = np.zeros(((99, 99)))
        img_stacked = np.stack([img_geometry, img_null])
        img_stacked = np.concatenate([img_stacked, img_damage])
        tensor = torch.tensor(img_stacked, dtype=torch.float)
        transform = transforms.RandomHorizontalFlip(p=0.5)
        tensor = transform(tensor)
        k = np.random.rand()
        roll = np.random.randint(self.img_res)
        tensor = torch.roll(tensor, roll, -1)
        return (tensor[0].view(1, self.img_res, self.img_res), #geometry
                tensor[1:], #damage
                self.imp_shrinkage, #shrinkage
                torch.tensor([sequence['obs_shrinkage']], dtype=torch.float).flatten(),
                torch.tensor([sequence['stiffness']], dtype=torch.float).flatten())