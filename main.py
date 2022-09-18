import sys
import rlg
from egg import core
import torch
def main(args):

    core.util.init() # most important line of code

    vision = rlg.Vision()
    class_prediction = rlg.PretrainVision(vision)
    class_prediction.load_state_dict(torch.load('./models/class_prediction.pth', map_location=torch.device('cpu')))
    # Agent's and game's setup
    sender = rlg.SenderCifar10(class_prediction.vision_module)
    receiver= rlg.ReceiverCifar10()

    game = rlg.LanguageGame(sender, receiver)#, sender_entropy_coeff=0.005, receiver_entropy_coeff=0.000)
  #  game.load_state_dict(torch.load('./models/reinf_Own_Game_230_epochs_256_128.pth'))
    train_data_loader1,train_data_loader2,train_data_loader3, test_data_loader = rlg.load_dataset()

    # train for 3*20 epochs and automatically save resulting model in default way
    game.train2(20, train_data_loader1, test_data_loader)
    del(train_data_loader1)
    game.train2(20, train_data_loader2, test_data_loader)
    del (train_data_loader2)
    game.train2(20, train_data_loader3, test_data_loader)
    EntropyLogger.entropy_log

   # game.plot(test_data_loader)

print('hello')
if __name__ == "__main__":
    main(sys.argv)
