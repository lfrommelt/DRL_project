import re
import os
from tqdm import tqdm
import numpy as np
from PIL.Image import open as im_open
import torch
from torch.utils.data.sampler import SubsetRandomSampler

def load_dataset():
    '''
    Load train and test dataloaders from ./data
    We use 30000 train images and 5000 test images
    We have to split the train data in 3 parts, because of insufficient computational power (46Gb Ram) during training
    '''
    random_seed = 42
    custom_train_data = {}
    files = [file for file in os.listdir("data/continuous/")
            if re.search(r'.*\.jpg', file)]

    # Load the train and test dataset and normalize the images
    dataset_size = len(files)
    for i in tqdm(range(dataset_size)):
        file = files[i]
        im = np.moveaxis(
            np.array(im_open("data/continuous/" + file)), 2,
            0) /255.
        y = [float(value) for value in file[:-4].split(sep=', ')]
        custom_train_data[i] = (torch.tensor(im).to(dtype=torch.float32), torch.tensor(y))

    # Creating data indices for training and validation splits:
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[10000:], indices[:5000]
    train_indices1 = train_indices[:10000]
    train_indices2 = train_indices[10000:20000]
    train_indices3 = train_indices[20000:30000]
    train_sampler1 = SubsetRandomSampler(train_indices1)
    train_sampler2 = SubsetRandomSampler(train_indices2)
    train_sampler3 = SubsetRandomSampler(train_indices3)

    test_sampler = SubsetRandomSampler(test_indices)

    train_loader1 = torch.utils.data.DataLoader(custom_train_data, batch_size=32,
                                               sampler=train_sampler1)
    train_loader2 = torch.utils.data.DataLoader(custom_train_data, batch_size=32,
                                               sampler=train_sampler2)
    train_loader3 = torch.utils.data.DataLoader(custom_train_data, batch_size=32,
                                               sampler=train_sampler3)
    test_loader = torch.utils.data.DataLoader(custom_train_data, batch_size=32,
                                                    sampler=test_sampler)
    del(custom_train_data)
    del(train_sampler1)
    del(train_sampler2)
    del(train_sampler3)
    del(test_sampler)

    return train_loader1,train_loader2,train_loader3, test_loader
