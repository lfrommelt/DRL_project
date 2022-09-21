from rlg.architectures import *
from torch.nn import Module
from torch.optim import Adam
from torch import save
import torch
from egg.core import Interaction
import matplotlib.pyplot as plt


class EntropyLogger(core.Callback):
    '''
    Callback for logging the entropies of the senders probability distributions,
    from which the messages are sampled during training. Values will be saved
    in a list as well as appended to a text file.
    '''
    entropy_log = []
    accumulated = []

    def __init__(self, reinforce=True):
        super().__init__()
        self.reinforce = reinforce

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        EntropyLogger.entropy_log.append(logs.aux['sender_entropy'].mean())
        with open('log_gs_entropy_200_128.txt', 'a') as file:
            print(EntropyLogger.entropy_log[-1], file=file)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        with open('log_gs_test_loss_200_128.txt', 'a') as file:
            print(loss, file=file)
        EntropyLogger.accumulated = []
    

class LanguageGame(core.SenderReceiverRnnReinforce):
    '''
    Class for the whole language game
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
        
        # callbacks will be activated during all relevant steps
        self.callbacks = []
        self.callbacks.append(EntropyLogger())
        self.callbacks.append(core.ConsoleLogger(as_json=False, print_train_loss=True))
        self.interaction_log = []

        loss = Hyperparameters.loss

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
        test_data (torch.DataLoader)
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

        game = LanguageGame(sender, receiver, sender_entropy_coeff=sender_entropy_coeff, receiver_entropy_coeff=receiver_entropy_coeff)
        game.load_state_dict(
            torch.load('./models/'+name+'.pth', map_location=core.get_opts().device))
        game.eval()
        return game

    def get_output(self, _, test_dataset):
        '''Get Receiver output'''
        interaction = \
            core.dump_interactions(self, test_dataset, gs=False, variable_length=True)

        print(interaction.message)
        plots = []
        titles = []
        for z in range(8):
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
      #  plt.savefig(name + '_train.png')


