from rlg.loss import *
import torch.nn as nn
import torch
from egg import core
from os.path import normpath
import os

class Hyperparameters:

    # loss for training
    loss = custom_loss
    # hidden size of the GRUs
    hidden_size = 256
    # embedding of discrete symbols
    emb_size = 128
    # number of possible symbols, including eos
    vocab_size = 4
    # maximal number of symbols per message
    max_len = 5


class Vision(nn.Module):
    '''
    Vision Module CNN architecture
    '''
    def __init__(self):
        super(Vision, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.activate1 = nn.LeakyReLU(inplace=True)
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(8)
        self.activate2 = nn.LeakyReLU(inplace=True)
        self.max2 = nn.MaxPool2d(4, 2)
        self.faltten1 = nn.Flatten()
        self.lin1 = nn.Linear(4232, 2500)
        self.activate3 = nn.LeakyReLU(inplace=True)
        self.lin2 = nn.Linear(2500, 500)
        self.activate4 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activate1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.activate2(x)
        x = self.max2(x)
        x = self.faltten1(x)
        x = self.lin1(x)
        x = self.activate3(x)
        x = self.lin2(x)
        x = self.activate4(x)

        return x

# Agent's vision module
class PretrainVision(nn.Module):
    '''
    Class for pretraining the vision module
    '''
    def __init__(self, vision_module):
        super(PretrainVision, self).__init__()
        self.vision_module = vision_module
        self.fc = nn.Linear(500, 6)

    def forward(self, x):
        x = self.vision_module(x)
        x = self.fc(torch.sigmoid(x))
        return x

    # function to check the accuracy of the vision-modules
    def check_accuracy(self, data_loader):
        '''
        evaluate the performance on a given dataset (torch.DataLoader)
        '''
        losses = 0
        num_samples = 0
        model = self
        model.eval()

        device = core.get_opts().device


        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device=device)
                y = y.to(device=device)
                score = model(x)
                loss = nn.functional.l1_loss(score, y)
                losses += loss.mean().item()
                num_samples += 1

        mean_loss = losses/num_samples
        return mean_loss

    def save(self, name):
        '''
        Save a trainable vision module to /models, training can be continued
        
        params:
        name (string): name of the file for saving
        '''
        torch.save(self.state_dict(), normpath('models/'+name+'.trainable_vision'))

# Receiver's and Sender's architecture
class Sender(nn.Module):
    '''
    This class constitutes an "agent" in EGG's terminology. It consists of a
    vision module and a layer, that maps to the hidden size of the sender of 
    the GRU
    '''
    def __init__(self, vision, output_size=Hyperparameters.hidden_size):
        super(Sender, self).__init__()
        self.fc = nn.Linear(500, output_size)
        self.vision = vision

    def forward(self, x, aux_input=None):
        with torch.no_grad():
            x = self.vision(x)
        x = self.fc(x)
        return x

class Receiver(nn.Module):
    '''
    This class constitutes an "agent" in EGG's terminology. It consists of a
    layer, that maps from the hidden size of the sender to the output size
    i.e. the size of the images
    '''
    def __init__(self, input_size=Hyperparameters.hidden_size):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(input_size, 3 * 100 * 100)

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        x = self.fc(channel_input)
        return torch.sigmoid(x)
