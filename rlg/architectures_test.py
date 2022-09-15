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


# activations
class VisionTest1(nn.Module):
    def __init__(self):
        super(VisionTest1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.activate1 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32,64 , kernel_size=3, stride=1, padding=1)
        self.activate2 = nn.ReLU(inplace=True)
        self.max2 = nn.AvgPool2d((100, 100), stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate1(x)
        x = self.conv3(x)
        x = self.activate2(x)
        x = self.max2(x)
        x = torch.squeeze(x)

        return x

    def save(self, name):
        torch.save(self.state_dict(), normpath('/models/'+name+'.vision'))

    @staticmethod
    def load(name="vision"):
        # todo: load weights from /models.
        vision = VisionTest()
        return vision

# Agent's vision module
class PretrainVisionTest1(nn.Module):
    def __init__(self, vision_module):
        super(PretrainVisionTest1, self).__init__()
        self.vision_module = vision_module
        self.fc3 = nn.Linear(64, 6)

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



# one-hot output
class VisionTest2(nn.Module):
    def __init__(self):
        super(VisionTest2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.activate1 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32,64 , kernel_size=3, stride=1, padding=1)
        self.activate2 = nn.ReLU(inplace=True)
        self.max2 = nn.AvgPool2d((100, 100), stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate1(x)
        x = self.conv3(x)
        x = self.activate2(x)
        x = self.max2(x)
        x = torch.squeeze(x)

        return x

    def save(self, name):
        torch.save(self.state_dict(), normpath('/models/'+name+'.vision'))

    @staticmethod
    def load(name="vision"):
        # todo: load weights from /models.
        vision = VisionTest()
        return vision

# Agent's vision module
class PretrainVisionTest2(nn.Module):
    def __init__(self, vision_module):
        super(PretrainVisionTest2, self).__init__()
        self.vision_module = vision_module
        self.fc3 = nn.Linear(64, 5)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.vision_module(x)
        x1 = self.fc3(torch.sigmoid(x))
        x2 = self.fc4(torch.softmax(x, dim = 0))
        return torch.concat((x1,x2),dim=1)

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



# max pool
class VisionTest3(nn.Module):
    def __init__(self):
        super(VisionTest3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.activate1 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32,64 , kernel_size=3, stride=1, padding=1)
        self.activate2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d((100, 100), stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate1(x)
        x = self.conv3(x)
        x = self.activate2(x)
        x = self.max2(x)
        x = torch.squeeze(x)

        return x

    def save(self, name):
        torch.save(self.state_dict(), normpath('/models/'+name+'.vision'))

    @staticmethod
    def load(name="vision"):
        # todo: load weights from /models.
        class_prediction = PretrainVision(vision)  # note that we pass vision - which we want to pretrain
        class_prediction = class_prediction.to(device)

        # class_prediction.load_state_dict(torch.load('Class_prediction_very_simple3.pth', map_location=torch.device('cpu')))

        vision = VisionTest()
        return vision

# Agent's vision module
class PretrainVisionTest3(nn.Module):
    def __init__(self, vision_module):
        super(PretrainVisionTest3, self).__init__()
        self.vision_module = vision_module
        self.fc3 = nn.Linear(64, 5)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.vision_module(x)
        x1 = self.fc3(torch.sigmoid(x))
        x2 = self.fc4(torch.softmax(x, dim = 0))
        return torch.concat((x1,x2),dim=1)

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

    @staticmethod
    def load(name="vision"):
        # todo: load weights from /models.

        device = core.get_opts().device
        vision = TestVision3()

        class_prediction = PretrainVisionTest3(vision)  # note that we pass vision - which we want to pretrain
        class_prediction = class_prediction.to(device)

        class_prediction.load_state_dict(torch.load(normpath('models/'+name+'.trainable_vision'), map_location=torch.device(device)))

        return class_prediction




# max pool
class VisionTest4(nn.Module):
    def __init__(self):
        super(VisionTest4, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.activate1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.activate2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32,64 , kernel_size=3, stride=1, padding=1)
        self.activate3 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d((100, 100), stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate1(x)
        x = self.conv2(x)
        x = self.activate2(x)
        x = self.conv3(x)
        x = self.activate3(x)
        x = self.max2(x)
        x = torch.squeeze(x)

        return x

    def save(self, name):
        torch.save(self.state_dict(), normpath('/models/'+name+'.vision'))

    @staticmethod
    def load(name="vision"):
        # todo: load weights from /models.
        vision = VisionTest()
        return vision

# Agent's vision module
class PretrainVisionTest4(nn.Module):
    def __init__(self, vision_module):
        super(PretrainVisionTest4, self).__init__()
        self.vision_module = vision_module
        self.fc3 = nn.Linear(64, 5)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.vision_module(x)
        x1 = self.fc3(torch.sigmoid(x))
        x2 = self.fc4(torch.softmax(x, dim = 0))
        return torch.concat((x1,x2),dim=1)

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