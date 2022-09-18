import egg.core as core
from egg.zoo.emcom_as_ssl.LARC import LARC

from rlg.architectures import *
#from rlg.interactions_fix import fix
#from rlg import FixedTrainer
from rlg.trainer_fix import FixedTrainer
from torch.nn import Module
from torch.optim import Adam
from torch import save
import torch
from egg.core import Interaction
from torch.distributions import Categorical
#import matplotlib.pyplot as plt

#import argparse

#from egg.core.util import get_opts

'''def egg_is_fucked_up():
    return {'validation_freq':10}

core.util.get_opts = egg_is_fucked_up'''

'''global common_opts
common_opts = {'validation_freq':10}'''
'''fix(core.dump_interactions)
fix(core.Trainer.__init__)'''


class EntropyLogger(core.Callback):
    entropy_log = []
    accumulated = []

    def __init__(self, reinforce=True):
        super().__init__()
        self.reinforce = reinforce

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if not self.reinforce:
            EntropyLogger.entropy_log.append(self.aggregate())
        else:
            EntropyLogger.entropy_log.append(logs.aux['sender_entropy'].mean())

        with open('log_gs_entropy_200_128.txt', 'a') as file:
            print(EntropyLogger.entropy_log[-1], file=file)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        with open('log_gs_test_loss_200_128.txt', 'a') as file:
            print(loss, file=file)
        EntropyLogger.accumulated = []

    def aggregate(self):
        entropies = torch.FloatTensor([Categorical(logits=torch.FloatTensor(logits)).entropy() for logits in EntropyLogger.accumulated])
        EntropyLogger.accumulated = []
        return torch.stack(entropies).flatten().mean()

def gs_logging_wrapper(self, x, aux_input=None):
    sequence, logits = _forward(self, x, aux_input=None)
    if self.training:
        for unbatched in logits:
            probs = torch.distributions.one_hot_categorical.OneHotCategorical(logits=unbatched).probs
            print("probs",probs)
            EntropyLogger.accumulated.append(probs)
    return sequence


def _forward(self, x, aux_input=None):
    prev_hidden = self.agent(x, aux_input)
    prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

    e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
    sequence = []
    logits = []

    for step in range(self.max_len):
        if isinstance(self.cell, torch.nn.LSTMCell):
            h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
        else:
            h_t = self.cell(e_t, prev_hidden)

        step_logits = self.hidden_to_output(h_t)
        x = core.gs_wrappers.gumbel_softmax_sample(
            step_logits, self.temperature, self.training, self.straight_through
        )
        logits.append(step_logits.detach())

        prev_hidden = h_t
        e_t = self.embedding(x)
        sequence.append(x)

    sequence = torch.stack(sequence).permute(1, 0, 2)

    eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
    eos[:, 0, 0] = 1
    sequence = torch.cat([sequence, eos], dim=1)

    return sequence, logits

class LanguageGame(core.SenderReceiverRnnReinforce):
    '''
    Definition of the whole language game
    todo: implement actor critics
    '''

    def __init__(
                self,
                sender: Module,
                receiver: Module,
                sender_entropy_coeff: float = 0.005,
                receiver_entropy_coeff: float = 0.0,
                length_cost: float = 0.0,
                ):

        # wrap architectures in egg
        sender = core.RnnSenderReinforce(sender, Hyperparameters.vocab_size, Hyperparameters.emb_size, Hyperparameters.hidden_size, cell="gru", max_len=Hyperparameters.max_len)
        receiver = core.RnnReceiverDeterministic(receiver, Hyperparameters.vocab_size, Hyperparameters.emb_size, Hyperparameters.hidden_size, cell="gru")
        self.callbacks = []
        self.callbacks.append(EntropyLogger())
        self.callbacks.append(core.ConsoleLogger(as_json=False, print_train_loss=True))

        # todo: could be done with entropy and so on as well
        loss = Hyperparameters.loss
        self.interaction_log = []

        super().__init__(
            sender,
            receiver,
            loss,
       #     sender_entropy_coeff,
        #    receiver_entropy_coeff,
            length_cost,
        )

        self.optimizer = Adam([
            {'params': sender.parameters(), 'lr': 1e-4},
            {'params': receiver.parameters(), 'lr': 1e-3}])
     #   self.optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)

        # Callbacks

        

    def train2(self, n_epochs, train_data, test_data, save_name='default'):
        # games trainer
        self.trainer = core.Trainer(game=self,
                                   optimizer=self.optimizer,
                                   train_data=train_data,
                                   validation_data=test_data,
                                   callbacks=self.callbacks
                                   )

        self.trainer.train(n_epochs)
        save(self.state_dict(), 'models/' + save_name + '.pth')


    def get_trainer(self, train_data, test_data):
        self.trainer = core.Trainer(game=self,
                                        optimizer=self.optimizer,
                                        train_data=train_data,
                                        validation_data=test_data,
                                        callbacks=self.callbacks
                                        )
        return self.trainer


    @staticmethod
    def load(name = 'default', sender_entropy_coeff=0.002, receiver_entropy_coeff=0.0005, vision_class = Vision):
        vision = vision_class()

        # Agent's and game's setup
        sender = SenderCifar10(vision)
        receiver = ReceiverCifar10()

        game = LanguageGame(sender, receiver, sender_entropy_coeff=sender_entropy_coeff, receiver_entropy_coeff=receiver_entropy_coeff)
        game.load_state_dict(
            torch.load('./models/'+name+'.pth', map_location=core.get_opts().device))
        game.eval()
        return game

    # Easy plotting of the images
    def get_output(self, _, test_dataset):
        interaction = \
            core.dump_interactions(self, test_dataset, gs=False, variable_length=True)

        print(interaction.message)
        plots = []
        titles = []
        for z in range(10):
            src = interaction.sender_input[z].permute(1, 2, 0)
            dst = interaction.receiver_output[z].view(3, 100, 100).permute(1, 2, 0)
            interaction_message = interaction.message[z]

            image = torch.cat([src, dst], dim=1).cpu().numpy()
            title = (f"Input: digit {z}, channel message {interaction_message}")
            plt.title = title
            plots.append(image)
            titles.append(title)

        return plots, titles

    def plot(self, test_data, name="reinf_Own_Game"):
        plots_game_lstm_Cifar10_GS, titles_game_lstm_Cifar10_GS = self.get_output(self, test_data)

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
            img = img
            title = titles[i - 1]
            fig.add_subplot(rows, columns, i)
            fig.tight_layout()
            plt.gca().set_title(title)
            plt.imshow(img)

        for i in fig.axes:
            i.set_xticks([])
            i.set_yticks([])

        plt.savefig(name + '_train.png')


