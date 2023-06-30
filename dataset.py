import os
import glob
from PIL import Image, ImageOps
import numpy as np
import torch
import csv
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, res, type=None):
        if type=='simple':
            self.imgs_path = "reduxed_results/geometry/"
            self.label_path = "reduxed_results/damage_fields/"
            input_path = [self.imgs_path + str(x) + '.npy' for x in range(15000)]
            output_path = [self.label_path + str(x) + '_99.npy' for x in range(15000)]
            self.data = []
            for i in range(15000):
                self.data.append([input_path[i], output_path[i]])
                
        else:
            raise('Not implemented')
        
        self.img_res = 99

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]
        transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        )       
        img = torch.tensor(np.load(img_path)[:-1,:-1], dtype=torch.float)
        label = torch.tensor(np.load(label_path)[:-1,:-1], dtype=torch.float)
        tensor = torch.stack([img, label])
        tensor = torch.unsqueeze(tensor,1)
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
        return tensor[0], tensor[1]


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