import sys
import rlg
from egg import core
import torch
from torchsummary import summary
import torch.nn.functional as F
import time
import re
import os
from tqdm import tqdm
import numpy as np
from PIL.Image import open as im_open
import torch

def load_relabeled_dataset(split=8/10):
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

    for i in tqdm(range(n_train)):
        file = files[i]
        im = np.moveaxis(
            np.array(im_open("data/continuous/" + file)), 2,
            0) /255.
        y = [float(value) for value in file[:-4].split(sep=', ')]
        y = relabel(y)
        custom_train_data[i] = (torch.tensor(im).to(dtype=torch.float32), torch.tensor(y))

    custom_test_data = {}
    test_files = os.listdir("data/continuous/")[n_train:n_train+n_test]  # [2:]
    for i in tqdm(range(len(test_files))):
        file = test_files[i]
        im = np.moveaxis(
            np.array(im_open("data/continuous/" + file)), 2,
            0) /255.
        y = [float(value) for value in file[:-4].split(sep=', ')]
        y = relabel(y)
        custom_test_data[i] = (torch.tensor(im).to(dtype=torch.float32), torch.tensor(y))
    # create the dataloaders
    train_data_loader = torch.utils.data.DataLoader(custom_train_data, batch_size=32, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(custom_test_data, batch_size=32, shuffle=True)
    del(custom_test_data)
    del(custom_train_data)

    return train_data_loader, test_data_loader

def relabel(target):
    # one-hot encoding for shapes (and outline?)
    (x,y,shape,size,color,outline) = target
    return [x,y,size,color,outline,
            1 if shape < 0.3 else 0,
            1 if shape > 0.3 and shape > 0.6 else 0,
            1 if shape > 0.6 else 0,]


def main(args):
    # todo: put hyperparams in args; probably we will not use main a lot, though
    core.util.init()

    device = core.get_opts().device

    #test model2
    class_prediction = PretrainVisionTest3.load('gap3')

    train_data_loader, test_data_loader = load_relabeled_dataset(8/10)

    print(class_prediction.check_accuracy(train_data_loader))
    print(class_prediction.check_accuracy(test_data_loader))



if __name__ == "__main__":
    main(sys.argv)