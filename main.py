import sys
import rlg
from egg import core
import torch
def main(args):

    core.util.init() # most important line of code

    # todo: should we directly load the vision module?
    # small vision module with few parameters (99.4 kB, not 47 MB), nice for testing things
    vision = rlg.Vision()
    class_prediction = rlg.PretrainVision(vision)
    class_prediction.load_state_dict(torch.load('./models/class_prediction.pth', map_location=torch.device('cpu')))
    # Agent's and game's setup
    sender = rlg.SenderCifar10(class_prediction.vision_module)
    receiver= rlg.ReceiverCifar10()

    game = rlg.LanguageGame(sender, receiver)#, sender_entropy_coeff=0.005, receiver_entropy_coeff=0.000)
  #  game.load_state_dict(torch.load('./models/reinf_Own_Game_230_epochs_256_128.pth'))
    train_data_loader, test_data_loader = rlg.load_dataset(0.8)
  #  game = rlg.LanguageGame.load(name="reinf_Own_Game_230_epochs_256_128")
    # train for 20 epochs and automatically save resulting model in default way
    game.train2(50, train_data_loader, test_data_loader)
   # game.plot(test_data_loader)
 #   print('Original test loss:', game.trainer.eval(test_data_loader)[0])

    # default loading needs no arguments at all
   # game2 = rlg.LanguageGame.load(sender_entropy_coeff=0.005, receiver_entropy_coeff=0.0005, vision_class=rlg.Vision)
   # game.train2(1,train_data_loader,test_data_loader)
    # this is why I know that egg loves us
    #game2.get_trainer(train_data_loader, test_data_loader)

    # kp, warum der loss anders ist, jedes mal wenn er neu lädt bleibt er gleich aber ist halt unterschiedlich zum ersten
    # (und gerne quatschwerte). Sämtliche geladenen weights sind allerings die gleichen und auch die outputs sind gleich
    # (bei gleichen inputs)
    #print('Reloaded test loss:', game2.trainer.eval(test_data_loader)[0])

print('hello')
if __name__ == "__main__":
    main(sys.argv)
