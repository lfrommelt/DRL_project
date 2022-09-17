import sys
import rlg
from egg import core

def main(args):
    # todo: put hyperparams in args; probably we will not use main a lot, though

    core.util.init()
    vision = rlg.PretrainVision.load().vision_module

    # Agent's and game's setup
    sender = rlg.SenderCifar10(vision)
    receiver= rlg.ReceiverCifar10()

    game = rlg.LanguageGame(sender, receiver, sender_entropy_coeff=0.002, receiver_entropy_coeff=0.0005)

    train_data_loader, test_data_loader = rlg.load_dataset((5,5))

    # train for 20 epochs
    # train2 because some weird namespace fuckups
    game.train2(1, train_data_loader, test_data_loader)


print('hello')
if __name__ == "__main__":
    main(sys.argv)
