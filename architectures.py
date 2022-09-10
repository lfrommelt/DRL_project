
class Hyperparameters():
    loss = mce_loss
    # todo: params

def mce_loss(y_true, y_pred):
    loss = np.mean((y_true-y_pred)**2)
    return loss

def custom_loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None, color_weight=0.8):
    """
    Custom loss function that weights the loss from colored pixels
    <-> white pixels from the original image 9:1
    """

    sender_input = sender_input.view([-1, 3 * 100 * 100])
    sender_input = sender_input.cpu().detach().numpy()
    receiver_output = receiver_output.cpu().detach().numpy()

    losses = np.zeros(len(sender_input))
    # fucking batching screws up vectorized numpy use >_<
    for i in range(len(sender_input)):
        a = sender_input[i]
        b = receiver_output[i]
        white = np.average(mce(a[a==1],b[a==1]))
        colour = np.average(mce(a[a<1],b[a<1]))
        losses[i] = color_weight*colour+(1-color_weight)*white

    return torch.from_numpy(losses), {}

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

        self.fc = nn.Linear(5000, 3 * 100 * 100)

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        x = self.fc0(channel_input)
        x = self.activate1(x)
        x = self.fc05(x)
        x = self.activate2(x)
        x = self.fc(x)
        return torch.sigmoid(x)

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
        self,
        sender,
        receiver,
        loss,
        sender_entropy_coeff,
        receiver_entropy_coeff,
        length_cost,
        baseline_type,
        train_logging_strategy,
        test_logging_strategy,
    )
    self.optimizer = torch.optim.Adam([
        {'params': game_lstm_Cifar10_GS.sender.parameters(), 'lr': 1e-4},
        {'params': game_lstm_Cifar10_GS.receiver.parameters(), 'lr': 1e-3}])
    optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)

    # Callbacks
    callbacks = []
    callbacks.append(core.ConsoleLogger(as_json=False, print_train_loss=True))

    # games trainer
    self.trainer = core.Trainer(game=game_lstm_Cifar10_GS,
                                           optimizer=optimizer,
                                           train_data=train_data_loader,
                                           validation_data=test_data_loader,
                                           callbacks=callbacks)
