import matplotlib.pyplot as plt
from pylab import rcParams
from matplotlib import image
from tqdm import tqdm
import os
import egg.core as core
from torchvision import  transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from egg.zoo.emcom_as_ssl.LARC import LARC

# start interactions fix
import egg.core as core
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch.distributed as distrib

from egg.core.batch import Batch
from egg.core import Interaction

def dump_interactions(
    game: torch.nn.Module,
    dataset: "torch.utils.data.DataLoader",
    gs: bool,
    variable_length: bool,
    device: Optional[torch.device] = None,
    apply_padding: bool = True,
) -> Interaction:
    """
    A tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param gs: whether the messages should be argmaxed over the last dimension.
        Handy, if Gumbel-Softmax relaxation was used for training.
    :param variable_length: whether variable-length communication is used.
    :param device: device (e.g. 'cuda') to be used.
    :return: The entire log of agent interactions, represented as an Interaction instance.
    """
    train_state = game.training  # persist so we restore it back
    game.eval()
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    full_interaction = None

    with torch.no_grad():
        for batch in dataset:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(device)
            _, interaction = game(*batch)
            interaction = interaction.to("cpu")

            if gs:
                interaction.message = interaction.message.argmax(
                    dim=-1
                )  # actual symbols instead of one-hot encoded
            if apply_padding and variable_length:
                assert interaction.message_length is not None
                for i in range(interaction.size):
                    length = interaction.message_length[i].long().item()
                    interaction.message[i, length:] = 0  # 0 is always EOS

            full_interaction = (
             #   full_interaction + interaction
             #   if full_interaction is not None
             #   else interaction
                full_interaction
		 )

    game.train(mode=train_state)
    return interaction


core.dump_interactions = dump_interactions
# end interactions fix

# Parameters and setup
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
transform = transforms.ToTensor()
rcParams['figure.figsize'] = 5, 10
# For convenince and reproducibility, we set some EGG-level command line arguments here
opts = core.init(params=['--random_seed=7',  # will initialize numpy, torch, and python RNGs
                         '--lr=1e-3',  # sets the learning rate for the selected optimizer
                         '--batch_size=32',
                         '--optimizer=adam'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# todo
# Easy plotting of the images
def plot(game, test_dataset, is_gs, variable_length, is_mnist):
    interaction = \
        core.dump_interactions(game, test_dataset, is_gs, variable_length)

    print(interaction.message)
    plots = []
    titles = []
    for z in range(10):
        if not is_mnist and not is_gs:
           # print(np.array(interaction.sender_input[z]).shape)
           # print(np.array(interaction.receiver_output[z]).shape)
            src = interaction.sender_input[z].permute(1, 2, 0)
            dst = interaction.receiver_output[z].view(3, 100, 100).permute(1, 2, 0)

        else:
            src = interaction.sender_input[z].permute(1, 2, 0)
            dst = interaction.receiver_output[z].view(-1, 3, 100, 100)
            dst = dst[-1]
            dst2 = dst[:-1]
            dst = dst.permute(1, 2, 0)
        if is_gs:
            interaction_message = interaction.message[z]
        elif not is_mnist and not is_gs:
            interaction_message = interaction.message[z]
        else:
            interaction_message = (
                f"Input: digit {z}, channel message tensor({torch.argmax(interaction.message[z], dim=1)})")

        image = torch.cat([src, dst], dim=1).cpu().numpy()
        title = (f"Input: digit {z}, channel message {interaction_message}")
        plt.title = title
        plots.append(image)
        titles.append(title)
    return plots, titles#, dst2

# function to check the accuracy of the vision-modules
def check_accuracy(loader, model):
    mean_loss = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            score = model(x)
            loss = F.l1_loss(score, y)
            mean_loss += loss.mean().item()
            num_samples += 1

        print('Got: ', mean_loss / num_samples)

def custom_loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
    """
    Custom loss function that weights the loss from colored pixels
    <-> white pixels from the original image 9:1
    """

    sender_input = sender_input.view([-1, 3 * 100 * 100])
    sender_input = sender_input.cpu().detach().numpy()
    receiver_output = receiver_output.cpu().detach().numpy()

    white = np.average(bce_loss(a[a==1],b[a==1]))
    colour = np.average(bce_loss(a[a<1],b[a<1]))
    return 0.8*colour+0.2*white

def bce_loss(target, output):
    loss = -sum([y * np.log(x) for y, x in zip((target, 1-target),(output, 1 - output))])
    return np.nan_to_num(loss)

def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1)
"""
def custom_loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
    """
 #   Custom loss function that weights the loss from colored pixels
  #  <-> white pixels from the original image 9:1
    """
    mae = nn.L1Loss()


    sender_input = sender_input.view([-1, 3 * 100 * 100])
    #default = mae(sender_input, receiver_output)
    sender_input = sender_input.cpu().detach().numpy()

    receiver_output = receiver_output.cpu().detach().numpy()
    losses_default = []
 #   print(sender_input)
  #  print(receiver_output)
    for i in range(len(sender_input)):
    #    loss = 0
     #   number_pixel = 0
    #    print(type(sender_input[i]))
     #   print(type(receiver_output[i]))
        loss = mae(torch.tensor(sender_input[i]),torch.tensor(receiver_output[i]))

        losses_default.append(loss)
    zeros = []
    colors = []
    for i in sender_input:
        zero = np.where(i == 1.)[0]
        zeros.append(zero)
        color = np.where(i != 1.)[0]
        colors.append(color)

    """
#    losses_white = []
 #   for i in tqdm(range(len(zeros))):
  #      loss = 0
   #     number_pixels = 0
    #    for j in zeros[i]:
     #       loss += BinaryCrossEntropy(sender_input[i][j],receiver_output[i][j])
      #      number_pixels += 1
       # losses_white.append(loss/number_pixels)
    """

    losses_color = []
    for i in range(len(colors)):
        loss = 0
        number_pixel = 0
        for j in colors[i]:
            loss += mae(torch.tensor(sender_input[i][j]), torch.tensor(receiver_output[i][j]))
            number_pixel += 1
        loss = loss/number_pixel
        losses_color.append(loss)

    #losses_weigthed = []
    #for i in range(len(losses_white)):
     #   loss = (losses_white[i]*0.1)+(losses_color[i]*0.9)
      #  losses_weigthed.append(loss)



    losses_weigted = []
    for i in range(len(losses_color)):
        loss = losses_default[i] + (losses_color[i]*5)
        loss = loss/6
        losses_weigted.append(loss)

 #   print(losses_color)
 #   print(losses_weigted)
 #   print(losses_default)
    #exit()
    return torch.tensor(losses_weigted), {}

# default loss function
def default_loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
  #  pic = sender_input.view([-1,3,100,100])
  #  pic1 = (pic[2].permute(1, 2, 0))*255
  #  plt.imshow(pic1)
  #  plt.show()
  #  exit()

    loss = F.binary_cross_entropy(receiver_output,
                                  sender_input.view([-1, 3*100*100]),
                                  reduction='none').mean(dim=1)
    return loss, {}
"""
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

# Load the train and test dataset and normalize the images
# todo: randomize with seed
n_train = 2000
n_test = 500
custom_train_data = {}
files = os.listdir("data/continuous/")  # [2:]
for i in tqdm(range(len(files))):
    file = files[i]
    im = np.moveaxis(
        image.imread("data/continuous/" + file), 2,
        0) /255.
    y = [float(value) for value in file[:-4].split(sep=', ')]
    custom_train_data[i] = (torch.tensor(im).to(dtype=torch.float32), torch.tensor(y))

custom_test_data = {}
test_files = os.listdir("data/continuous/")  # [2:]
for i in tqdm(range(len(test_files))):
    file = test_files[i]
    im = np.moveaxis(
        image.imread("data/continuous/" + file), 2,
        0) /255.
    y = [float(value) for value in file[:-4].split(sep=', ')]
    custom_test_data[i] = (torch.tensor(im).to(dtype=torch.float32), torch.tensor(y))

# Instantiate the vision module (CNN) and load a pretrained version
vision = Vision()
class_prediction = PretrainVision(vision)  # note that we pass vision - which we want to pretrain
optimizer = core.build_optimizer(class_prediction.parameters())  # uses command-line parameters we passed to core.init
class_prediction = class_prediction.to(device)
class_prediction.load_state_dict(torch.load('Class_prediction_very_simple3.pth', map_location=torch.device('cpu')))

# create the dataloaders
train_data_loader = torch.utils.data.DataLoader(custom_train_data, batch_size=32, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(custom_test_data, batch_size=32, shuffle=True)
del(custom_test_data)
del(custom_train_data)

# Game's parameter
hidden_size = 64
emb_size = 32
vocab_size = 6

# Receiver's and Sender's architecture
class SenderCifar10(nn.Module):
    def __init__(self, vision, output_size):
        super(SenderCifar10, self).__init__()
        self.fc = nn.Linear(500, output_size)
        self.vision = vision

    def forward(self, x, aux_input=None):
        with torch.no_grad():
            x = self.vision(x)
        x = self.fc(x)
        return x

class ReceiverCifar10(nn.Module):
    def __init__(self, input_size):
        super(ReceiverCifar10, self).__init__()
        self.fc0 = nn.Linear(input_size, 1000)
        self.activate1 = nn.LeakyReLU()

        self.fc05 = nn.Linear(1000, 5000)
        self.activate2 = nn.LeakyReLU()
        #self.fc06 = nn.Linear(3000,20000)
        #self.activate4 = nn.LeakyReLU()
     #   self.fc07 = nn.Linear(10000, 20000)
      #  self.activate3 = nn.LeakyReLU()

        self.fc = nn.Linear(5000, 3 * 100 * 100)

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        x = self.fc0(channel_input)
        x = self.activate1(x)
        x = self.fc05(x)
        x = self.activate2(x)
       # x = self.fc06(x)
        #x = self.activate4(x)
        # = self.fc07(x)
        #x = self.activate3(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# Agent's and game's setup
senderCifar10 = SenderCifar10(class_prediction.vision_module, hidden_size)
receiverCifar10 = ReceiverCifar10(hidden_size)

sender_lstm_Cifar10_GS = core.RnnSenderReinforce(senderCifar10, vocab_size, emb_size,hidden_size,
                                   cell="gru", max_len=10)

receiver_lstm_Cifar10_GS = core.RnnReceiverDeterministic(receiverCifar10, vocab_size, emb_size,
                    hidden_size, cell="gru")

game_lstm_Cifar10_GS = core.SenderReceiverRnnReinforce(sender_lstm_Cifar10_GS,
                                                receiver_lstm_Cifar10_GS,
                                                custom_loss, sender_entropy_coeff=0.002, receiver_entropy_coeff=0.0005)

# Optimizer definition
optimizer = torch.optim.Adam([
    {'params': game_lstm_Cifar10_GS.sender.parameters(), 'lr': 1e-4},
    {'params': game_lstm_Cifar10_GS.receiver.parameters(), 'lr': 1e-3}])
optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)

# Callbacks
callbacks = []
callbacks.append(core.ConsoleLogger(as_json=False, print_train_loss=True))

# games trainer
trainer_lstm_Cifar10_GS = core.Trainer(game=game_lstm_Cifar10_GS,
                                       optimizer=optimizer,
                                       train_data=train_data_loader,
                                       validation_data=test_data_loader,
                                       callbacks=callbacks)

#load an older version of the game
game_lstm_Cifar10_GS.load_state_dict(torch.load('models/reinf_Own_Game_completel_new_try2.pth',map_location=torch.device('cpu')))

# train the game
n_epochs = 1
tqdm(trainer_lstm_Cifar10_GS.train(n_epochs))

# save the games current status
torch.save(game_lstm_Cifar10_GS.state_dict(), 'models/reinf_Own_Game_completel_new_try2.pth')

# create plot of input and output images from train set
plots_game_lstm_Cifar10_GS, titles_game_lstm_Cifar10_GS= plot(game_lstm_Cifar10_GS, train_data_loader, is_gs=False,
                                                               variable_length=True, is_mnist=False)
"""
for i in dst2:
    img = i.permute(1, 2, 0)
    img = img*255
    plt.imshow(img)
    plt.savefig('reinf_Own_Game_train_other_outputs_' + str(i) + '.png')
"""
plots = []
titles = []
for i in range(10):
    plots.append(plots_game_lstm_Cifar10_GS[i])
    titles.append(titles_game_lstm_Cifar10_GS[i])

fig = plt.figure(figsize=(100, 100))
fig.tight_layout()
columns = 1
rows = 10
for i in range(1, columns * rows + 1):
    img = plots[i - 1]
    img = img #* 255
    title = titles[i - 1]
    fig.add_subplot(rows, columns, i)
    fig.tight_layout()
    plt.gca().set_title(title)
    plt.imshow(img)

for i in fig.axes:
    i.set_xticks([])
    i.set_yticks([])

plt.savefig('reinf_Own_Game_train.png')

# create plot of input and output images from test set
plots_game_lstm_Cifar10_GS, titles_game_lstm_Cifar10_GS = plot(game_lstm_Cifar10_GS, test_data_loader, is_gs=False,
                                                               variable_length=True, is_mnist=False)

plots = []
titles = []
for i in range(10):
    plots.append(plots_game_lstm_Cifar10_GS[i])
    titles.append(titles_game_lstm_Cifar10_GS[i])

fig = plt.figure(figsize=(100, 100))
fig.tight_layout()
columns = 1
rows = 10
for i in range(1, columns * rows + 1):
    img = plots[i - 1]
    img = img #* 255
    title = titles[i - 1]
    fig.add_subplot(rows, columns, i)
    fig.tight_layout()
    plt.gca().set_title(title)
    plt.imshow(img)

for i in fig.axes:
    i.set_xticks([])
    i.set_yticks([])
#torch.save(game_lstm_Cifar10_GS.state_dict(), '/home/ui/Downloads/reinf_Own_Game.pth')
plt.savefig('reinf_Own_Game_test.png')
