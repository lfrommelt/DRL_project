import egg.core as core
from interactions_fix import fix


def main(args):
    # todo: put hyperparams in args; probably we will not use main a lot, though
    vision = Vision.load()

    # Agent's and game's setup
    senderCifar10 = SenderCifar10(class_prediction.vision_module, hidden_size)
    receiverCifar10 = ReceiverCifar10(hidden_size)

    sender_lstm_Cifar10_GS = core.RnnSenderReinforce(senderCifar10, vocab_size, emb_size,hidden_size,
                                       cell="gru", max_len=10)

    receiver_lstm_Cifar10_GS = core.RnnReceiverDeterministic(receiverCifar10, vocab_size, emb_size,
                        hidden_size, cell="gru")

    game = Game(sender_lstm_Cifar10_GS,
                receiver_lstm_Cifar10_GS,
                custom_loss2, sender_entropy_coeff=0.002, receiver_entropy_coeff=0.0005
                )

    train_data_loader, test_data_loader = load_dataset

    # train for 20 epochs
    game.train(20, train_data, test_data)


print('hello')
if name == "__main__":
    main(sys.argv)
