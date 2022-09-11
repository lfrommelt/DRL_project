import egg.core as core


fix(core.dump_interactions)


class LanguageGame(SenderReceiverRnnReinforce):
    '''
    Definition of the whole language game
    todo: implement actor critics
    '''

    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        sender_entropy_coeff: float = 0.0,
        receiver_entropy_coeff: float = 0.0,
        length_cost: float = 0.0,
        baseline_type: Baseline = MeanBaseline,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
    super().__init__(
        sender,
        receiver,
        loss,
        sender_entropy_coeff,
        receiver_entropy_coeff,
        length_cost,
        baseline_type,
        train_logging_strategy,
        test_logging_strategy,
    ):
        self.optimizer = torch.optim.Adam([
            {'params': game_lstm_Cifar10_GS.sender.parameters(), 'lr': 1e-4},
            {'params': game_lstm_Cifar10_GS.receiver.parameters(), 'lr': 1e-3}])
        optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)

        # Callbacks
        callbacks = []
        callbacks.append(core.ConsoleLogger(as_json=False, print_train_loss=True))


    def train(self, n_epochs, train_data, test_data, save_file='models/reinf_Own_Game.pth'):
        # games trainer
        self.trainer = core.Trainer(game=game_lstm_Cifar10_GS,
                                   optimizer=optimizer,
                                   train_data=train_data_loader,
                                   validation_data=test_data_loader,
                                   callbacks=callbacks
                                   )

        self.trainer.train(n_epochs)
        torch.save(game_lstm_Cifar10_GS.state_dict(), 'models/reinf_Own_Game_completel_new_try2.pth')

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

    def plot(self, name = reinf_Own_Game):
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
