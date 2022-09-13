from rlg.loss import *
import torch.nn as nn
import torch
from egg import core
from os.path import normpath
import os

class Hyperparameters:
    loss = custom_loss
    # Game's parameter
    hidden_size = 64
    emb_size = 32
    vocab_size = 6
    max_len = 10
    # todo: params


# Simple CNN's architecture
class Vision(nn.Module):
    def __init__(self):
        super(Vision, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4,stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.activate1 = nn.LeakyReLU(inplace=True)
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32,8 , kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(8)
        self.activate3 = nn.LeakyReLU(inplace=True)
        self.max2 = nn.MaxPool2d(4, 2)
        self.faltten1 = nn.Flatten()
        self.lin1 = nn.Linear(4232, 500)
        self.activate7 = nn.LeakyReLU(inplace=True)
        self.lin2 = nn.Linear(500,6)
        self.activate8 = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activate1(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = self.batchNorm2(x)
        x = self.activate3(x)
        x = self.max2(x)
        x = self.faltten1(x)
        x = self.lin1(x)
        x = self.activate7(x)

        return x

    def save(self, name):
        torch.save(self.state_dict(), normpath('/models/'+name+'.vision'))

    @staticmethod
    def load(name="vision"):
        # todo: load weights from /models.
        vision = Vision()
        return vision

# Agent's vision module
class PretrainVision(nn.Module):
    def __init__(self, vision_module):
        super(PretrainVision, self).__init__()
        self.vision_module = vision_module
        self.fc3 = nn.Linear(500, 6)

    def forward(self, x):
        x = self.vision_module(x)
        x = self.fc3(torch.sigmoid(x))
        return x

    # function to check the accuracy of the vision-modules
    def check_accuracy(self, data_loader):
        mean_loss = 0
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
                mean_loss += loss.mean().item()
                num_samples += 1

            #print('Got: ', mean_loss / num_samples)
        return mean_loss

    def save(self, name):
        print(os.getcwd())
        torch.save(self.state_dict(), normpath('models/'+name+'.trainable_vision'))

# Receiver's and Sender's architecture
class SenderCifar10(nn.Module):
    def __init__(self, vision, output_size=Hyperparameters.hidden_size):
        super(SenderCifar10, self).__init__()
        self.fc = nn.Linear(500, output_size)
        self.vision = vision

    def forward(self, x, aux_input=None):
        with torch.no_grad():
            x = self.vision(x)
        x = self.fc(x)
        return x

class ReceiverCifar10(nn.Module):
    def __init__(self, input_size=Hyperparameters.hidden_size):
        super(ReceiverCifar10, self).__init__()
        self.fc0 = nn.Linear(input_size, 1000)
        self.activate1 = nn.LeakyReLU()

        self.fc05 = nn.Linear(1000, 5000)
        self.activate2 = nn.LeakyReLU()

        self.fc = nn.Linear(5000, 3 * 100 * 100)

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        x = self.fc0(channel_input)
        x = self.activate1(x)
        x = self.fc05(x)
        x = self.activate2(x)
        x = self.fc(x)
        return torch.sigmoid(x)
