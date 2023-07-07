import os
import glob
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import torch
import csv
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, type=None):
        self.imgs_path = "reduxed_results/geometry/"
        self.label_path = "reduxed_results/damage_fields/"
        self.data = []
        if type=='simple':
            self.simple = True
            input_path = [self.imgs_path + str(x) + '.npy' for x in range(15000)]
            output_path = [self.label_path + str(x) + '_99.npy' for x in range(15000)]
            for i in range(15000):
                self.data.append([input_path[i], output_path[i]])   
        else:
            self.simple = False
            for i in range(15000):
                input_1_path = self.imgs_path + str(i) + '.npy'
                input_2_path = 'null'
                output_path = self.label_path + str(i) + '_11.npy'
                self.data.append([input_1_path, input_2_path, output_path])  
                for j in range(9):
                    input_1_path = self.imgs_path + str(i) + '.npy'
                    input_2_path = self.label_path + str(i) + '_' + str((j+1)*11) + '.npy'
                    output_path = self.label_path + str(i) + '_' + str((j+2)*11) + '.npy'
                    self.data.append([input_1_path, input_2_path, output_path])      
        self.img_res = 99

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.simple == True:
            img_path, label_path = self.data[idx]
            img = torch.tensor(np.load(img_path)[:-1,:-1], dtype=torch.float)
            label = torch.tensor(np.load(label_path)[:-1,:-1], dtype=torch.float)
            tensor = torch.stack([img, label])
            tensor = torch.unsqueeze(tensor,1)     
        else:
            img_path_1, img_path_2, label_path = self.data[idx]
            img_1 = torch.tensor(np.load(img_path_1)[:-1,:-1], dtype=torch.float)
            if img_path_2 == 'null':
                img_2 = torch.zeros((99, 99))
            else:
                img_2 = torch.tensor(np.load(img_path_2)[:-1,:-1], dtype=torch.float)
            label = torch.tensor(np.load(label_path)[:-1,:-1], dtype=torch.float)
            tensor = torch.stack([img_1, img_2, label])   
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
        return tensor[0:-1], tensor[-1]
    

class dataset_seq(Dataset):
    def __init__(self, type=None):
        self.imgs_path = "reduxed_results/geometry/"
        self.label_path = "reduxed_results/damage_fields/"
        self.img_res = 99
        imp_shrinkage_values = pd.read_csv(self.label_path + 'stiffness_0.csv', sep='\t', usecols=['#shr_imposed[-]']).values.tolist()
        imp_shrinkage_matrices = [np.full((self.img_res, self.img_res), value) for value in imp_shrinkage_values]
        imp_shrinkage_matrices_stacked = np.stack(imp_shrinkage_matrices)
        self.imp_shrinkage = torch.tensor(imp_shrinkage_matrices_stacked, dtype=torch.float)
        self.data = []
        for i in range(15000):
            if i != 5000:
                sequence = {}
                sequence['geometry'] = self.imgs_path + str(i) + '.npy'
                sequence['damage'] = []
                for j in range(10):
                    input = self.label_path + str(i) + '_' + str((j+1)*11) + '.npy'
                    sequence['damage'].append(input) 
                sequence['obs_shrinkage'] = pd.read_csv(self.label_path + f'stiffness_{i}.csv', sep='\t', usecols=['#shr_observed[-]']).values.tolist()
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
        return tensor[0].view(1,self.img_res,self.img_res), tensor[1:], self.imp_shrinkage, torch.tensor([sequence['obs_shrinkage']], dtype=torch.float).flatten()

def get_loaders(data, batch_size):
    n_train = int(0.8 * data.__len__())
    n_test = (data.__len__() - n_train) // 2
    n_val = data.__len__() - n_train - n_test
    torch.manual_seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(data, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size = batch_size)
    test_loader = DataLoader(test_set, batch_size = n_test, shuffle=True)
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