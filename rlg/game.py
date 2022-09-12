import egg.core as core
from egg.zoo.emcom_as_ssl.LARC import LARC

from rlg.architectures import Hyperparameters
from rlg.interactions_fix import fix
#from rlg import FixedTrainer
from rlg.trainer_fix import FixedTrainer
from torch.nn import Module
from torch.optim import Adam
from torch import save

#import argparse

#from egg.core.util import get_opts

'''def egg_is_fucked_up():
    return {'validation_freq':10}

core.util.get_opts = egg_is_fucked_up'''

'''global common_opts
common_opts = {'validation_freq':10}'''
'''fix(core.dump_interactions)
fix(core.Trainer.__init__)'''


class LanguageGame(core.SenderReceiverRnnReinforce):
    '''
    Definition of the whole language game
    todo: implement actor critics
    '''

    def __init__(
                self,
                sender: Module,
                receiver: Module,
                sender_entropy_coeff: float = 0.0,
                receiver_entropy_coeff: float = 0.0,
                length_cost: float = 0.0,
                ):

        # wrap architectures in egg
        sender = core.RnnSenderReinforce(sender, Hyperparameters.vocab_size, Hyperparameters.emb_size, Hyperparameters.hidden_size,cell="gru", max_len=Hyperparameters.max_len)
        receiver = core.RnnReceiverDeterministic(receiver, Hyperparameters.vocab_size, Hyperparameters.emb_size, Hyperparameters.hidden_size,cell="gru")

        # todo: could be done with entropy and so on as well
        loss = Hyperparameters.loss

        super().__init__(
            sender,
            receiver,
            loss,
            sender_entropy_coeff,
            receiver_entropy_coeff,
            length_cost,
        )
        optimizer = Adam([
            {'params': sender.parameters(), 'lr': 1e-4},
            {'params': receiver.parameters(), 'lr': 1e-3}])
        self.optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)

        # Callbacks
        self.callbacks = []
        self.callbacks.append(core.ConsoleLogger(as_json=False, print_train_loss=True))


    def train2(self, n_epochs, train_data, test_data, save_file='models/reinf_Own_Game.pth'):
        # games trainer
        self.trainer = core.Trainer(game=self,
                                   optimizer=self.optimizer,
                                   train_data=train_data,
                                   validation_data=test_data,
                                   callbacks=self.callbacks
                                   )

        self.trainer.train(n_epochs)
        save(self.state_dict(), 'models/reinf_Own_Game.pth')

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
        return plots, titles

    def plot(self, name = "reinf_Own_Game"):
        plots_game_lstm_Cifar10_GS, titles_game_lstm_Cifar10_GS= _plot(game_lstm_Cifar10_GS, train_data_loader, is_gs=False,
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

        plt.savefig(name+'_train.png')

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
        plt.savefig(name+'_test.png')
