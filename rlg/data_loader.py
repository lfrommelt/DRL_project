import re
import os
from tqdm import tqdm
import numpy as np
from PIL.Image import open as im_open
import torch

def load_dataset(split=8/10):
    '''
    Load train and test dataloaders from ./data
    Parameters:
    split -- ratio of testdata/absolute or (n_train, n_test)
    '''
    custom_train_data = {}
    files = [file for file in os.listdir("data/continuous/")
            if re.search(r'.*\.jpg', file)]
    try:
        n_train = split[0]
        n_test = split[1]
    except TypeError:
        n_train = int(len(files)*split)
        n_test = int(len(files)*(1-split))
    # Load the train and test dataset and normalize the images
    # todo: randomize with seed
    n_train = 2000
    n_test = 500
  # [2:]
    for i in tqdm(range(n_train)):
        file = files[i]
        im = np.moveaxis(
            np.array(im_open("data/continuous/" + file)), 2,
            0) /255.
        y = [float(value) for value in file[:-4].split(sep=', ')]
        custom_train_data[i] = (torch.tensor(im).to(dtype=torch.float32), torch.tensor(y))

    custom_test_data = {}
    test_files = os.listdir("data/continuous/")[n_train:n_train+n_test]  # [2:]
    for i in tqdm(range(len(test_files))):
        file = test_files[i]
        im = np.moveaxis(
            np.array(im_open("data/continuous/" + file)), 2,
            0) /255.
        y = [float(value) for value in file[:-4].split(sep=', ')]
        custom_test_data[i] = (torch.tensor(im).to(dtype=torch.float32), torch.tensor(y))
    # create the dataloaders
    train_data_loader = torch.utils.data.DataLoader(custom_train_data, batch_size=32, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(custom_test_data, batch_size=32, shuffle=True)
    del(custom_test_data)
    del(custom_train_data)

    return train_data_loader, test_data_loader
