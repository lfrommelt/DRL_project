# This script implements an alternative version of the game using Gumbel Softmax
# as a comparison to REINFORCE. Differences are almost exclusively in the
# different logging strategy

from rlg.architectures import *
from torch.nn import Module
from torch.optim import Adam
from torch import save
import torch
from egg.core import Interaction
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class EntropyLoggerGS(core.Callback):
    '''
    Callback for logging the entropies of the senders probability distributions,
    from which the messages are sampled during training. Values will be saved
    in a list as well as appended to a text file.
    '''
    entropy_log = []
    accumulated = []

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        EntropyLoggerGS.entropy_log.append(self.aggregate())

        with open('log_gs_entropy_200_128.txt', 'a') as file:
            print(EntropyLoggerGS.entropy_log[-1], file=file)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        with open('log_gs_test_loss_200_128.txt', 'a') as file:
            print(loss, file=file)
        EntropyLoggerGS.accumulated = []

    def aggregate(self):
        '''helper for getting averages'''
        entropies = ([Categorical(logits=torch.FloatTensor(logits)).entropy() for logits in EntropyLoggerGS.accumulated])
        EntropyLoggerGS.accumulated = []
        return torch.cat(entropies).flatten().mean()

def gs_logging_wrapper(self, x, aux_input=None):
    '''Wrapper for the forward step, in order to catch logits'''
    sequence, logits = _forward(self, x, aux_input=None)
    if self.training:
        for unbatched in logits:
            probs = torch.distributions.one_hot_categorical.OneHotCategorical(logits=unbatched).probs
            EntropyLoggerGS.accumulated.append(probs)

    return sequence


def _forward(self, x, aux_input=None):
    '''
    Changed version of forward() from 
    egg.core.reinforce_wrappers.RnnSenderGS
    for logging purposes
    '''
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

class LanguageGameGS(core.SenderReceiverRnnGS):
    '''
    Class for the whole langage game
    '''


    def __init__(
            self,
            sender: Module,
            receiver: Module,
            length_cost: float = 0.0,
    ):
        self.callbacks = []


        core.RnnSenderGS.forward = gs_logging_wrapper
        sender = core.RnnSenderGS(sender, vocab_size=Hyperparameters.vocab_size,
                                  embed_dim=Hyperparameters.emb_size,
                                  hidden_size=Hyperparameters.hidden_size, cell="gru",
                                  max_len=Hyperparameters.max_len, temperature=10)
        receiver = core.RnnReceiverGS(receiver, vocab_size=Hyperparameters.vocab_size,
                                      embed_dim=Hyperparameters.emb_size,
                                      hidden_size=Hyperparameters.hidden_size,
                                      cell="gru")
        self.callbacks.append(core.TemperatureUpdater(sender, minimum=0.8, update_frequency=1, decay=0.9))
        self.callbacks.append(EntropyLoggerGS())
        self.callbacks.append(core.ConsoleLogger(as_json=False, print_train_loss=True))

        # todo: could be done with entropy and so on as well
        loss = Hyperparameters.loss
        self.interaction_log = []

        super().__init__(
            sender,
            receiver,
            loss,
            length_cost,
        )

        self.optimizer = Adam([
            {'params': sender.parameters(), 'lr': 1e-5},
            {'params': receiver.parameters(), 'lr': 1e-4}])

    def train2(self, n_epochs, train_data, test_data, save_name='default'):
        '''
        Train the game for an amount of n_epochs with train_data.
        
        params:
        n_epochs (int)
        train_data (torch.DataLoader)
        test_data: Used only as an argument for trainer, no real impact
        save_name (str): file name for serializing models during training
        '''
        self.trainer = core.Trainer(game=self,
                                   optimizer=self.optimizer,
                                   train_data=train_data,
                                   validation_data=test_data,
                                   callbacks=self.callbacks
                                   )

        self.trainer.train(n_epochs)
        save(self.state_dict(), 'models/' + save_name + '.pth')


    def get_trainer(self, train_data, test_data):
        '''
        Get a trainer instance of a model for accessing trainer methods, like
        trainer.eval()
        '''
        self.trainer = core.Trainer(game=self,
                                        optimizer=self.optimizer,
                                        train_data=train_data,
                                        validation_data=test_data,
                                        callbacks=self.callbacks
                                        )
        return self.trainer


    @staticmethod
    def load(name = 'default', sender_entropy_coeff=0.002, receiver_entropy_coeff=0.0005, vision_class = Vision):
        '''
        Load and return a serialized game
        
        params:
        name (str): filename
        sender_entropy_coeff, receiver entropy_coeff (float): should be the same
            as during training
        vision_class (dType): In case a different vision architecture was used
        '''
        vision = vision_class()

        # Agent's and game's setup
        sender = Sender(vision)
        receiver = Receiver()

        game = LanguageGameGS(sender, receiver)
        game.load_state_dict(
            torch.load('./models/'+name+'.pth', map_location=core.get_opts().device))
        game.eval()
        return game

    def get_output(self, _, test_dataset):
        '''Get Receiver output'''
        interaction = \
            core.dump_interactions(self, test_dataset, gs=True, variable_length=True)

        plots = []
        titles = []
        for z in range(8):
            src = interaction.sender_input[z].permute(1, 2, 0)
            dst = interaction.receiver_output[z][0].view(3, 100, 100).permute(1, 2, 0)
            interaction_message = interaction.message[z]
            image = torch.cat([src, dst], dim=1).cpu().numpy()
            title = (f"Input: digit {z}, channel message {interaction_message}")
            plt.title = title
            plots.append(image)
            titles.append(title)

        return plots, titles

    def plot(self, test_data, name="reinf_Own_Game"):
        '''Easy plotting of examples'''
        plots_game_lstm_Cifar10_GS, titles_game_lstm_Cifar10_GS = self.get_output(self, test_data)

        plots = []
        titles = []
        for i in range(8):
            plots.append(plots_game_lstm_Cifar10_GS[i])
            titles.append(titles_game_lstm_Cifar10_GS[i])

        fig = plt.figure(figsize=(100, 100))
        fig.tight_layout()
        columns = 1
        rows = 8
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

        plt.show()
        plt.savefig(name + '_train.png')
