import re
import os
from tqdm import tqdm
import numpy as np
from PIL.Image import open as im_open
import torch

def load_dataset(split=8/10, batch_size=32, shuffle=True, random_seed=1):
    '''
    Load train and test dataloaders from ./data
    Parameters:
    split -- ratio of testdata/absolute as float or (n_train, n_test)
    '''
    custom_train_data = {}
    files = [file for file in os.listdir("data/continuous/")
            if re.search(r'.*\.jpg', file)]
    
    # very Pythonic, indeed
    try:
        n_train = split[0]
        n_test = split[1]
    except TypeError:
        n_train = int(len(files)*split)
        n_test = int(len(files)*(1-split))        
    
    dataset_size = len(files)
    
    # Creating data indices for training and validation splits:
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:n_train], indices[n_train:n_train+n_test]
    
    custom_train_data = []
    custom_test_data = []
    
    # wait until train set is loaded...
    for i in tqdm(train_indices):
        file = files[i]
        im = np.moveaxis(
            np.array(im_open("data/continuous/" + file)), 2,
            0) /255.
        y = [float(value) for value in file[:-4].split(sep=', ')]
        custom_train_data.append((torch.tensor(im).to(dtype=torch.float32), torch.tensor(y)))
        
    # wait until test set is loaded...
    for i in tqdm(test_indices):
        file = files[i]
        im = np.moveaxis(
            np.array(im_open("data/continuous/" + file)), 2,
            0) /255.
        y = [float(value) for value in file[:-4].split(sep=', ')]
        custom_test_data.append((torch.tensor(im).to(dtype=torch.float32), torch.tensor(y)))
              
    # create the dataloaders
    train_data_loader = torch.utils.data.DataLoader(custom_train_data, batch_size=batch_size, shuffle=shuffle)
    test_data_loader = torch.utils.data.DataLoader(custom_test_data, batch_size=batch_size, shuffle=shuffle)
    
    # With memory intensive training, better not trust the garbage collector
    del(custom_test_data)
    del(custom_train_data)

    return train_data_loader, test_data_loader