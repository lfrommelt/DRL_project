import sys
import rlg
from egg import core

def main(args):

    core.util.init() # most important line of code

    # todo: should we directly load the vision module?
    # small vision module with few parameters (99.4 kB, not 47 MB), nice for testing things
    vision = rlg.PretrainVisionTest3.load('gap3.trainable_vision').vision_module

    # Agent's and game's setup
    sender = rlg.SenderCifar10(vision)
    receiver= rlg.ReceiverCifar10()

    game = rlg.LanguageGame(sender, receiver, sender_entropy_coeff=0.002, receiver_entropy_coeff=0.0005)

    train_data_loader, test_data_loader = rlg.load_dataset((5,5))

    # train for 20 epochs and automatically save resulting model in default way
    game.train2(1, train_data_loader, test_data_loader)
    print('Original test loss:', game.trainer.eval(test_data_loader)[0])

    # default loading needs no arguments at all
    game2 = rlg.LanguageGame.load(sender_entropy_coeff=0.002, receiver_entropy_coeff=0.0005, vision_class=rlg.VisionTest3)

    # this is why I know that egg loves us
    game2.get_trainer(train_data_loader, test_data_loader)

    # kp, warum der loss anders ist, jedes mal wenn er neu lädt bleibt er gleich aber ist halt unterschiedlich zum ersten
    # (und gerne quatschwerte). Sämtliche geladenen weights sind allerings die gleichen und auch die outputs sind gleich
    # (bei gleichen inputs)
    print('Reloaded test loss:', game2.trainer.eval(test_data_loader)[0])

print('hello')
if __name__ == "__main__":
    main(sys.argv)
