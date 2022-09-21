import sys
import rlg
from egg import core
import torch

def main(args):

    # initialize egg
    core.util.init()

    device = core.get_opts().device

    # initialize vision module
    class_prediction = rlg.PretrainVision(rlg.Vision())
    optimizer = core.build_optimizer(class_prediction.parameters())
    class_prediction = class_prediction.to(device)

    '''80% of data will be used for training. Alternativly, give a tuple.
    E.g. (100, 20) for 100 training and 20 test images.'''
    train_data_loader, test_data_loader = rlg.load_dataset((1/10))#0.8)

    for epoch in range(15):
        mean_loss, n_batches = 0, 0
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = class_prediction(data)
            loss = torch.nn.functional.l1_loss(output, target)
            loss.backward()
            optimizer.step()
            mean_loss += loss.mean().item()
            n_batches += 1

        print(f'Train Epoch: {epoch}, mean loss: {mean_loss / n_batches}')

    class_prediction.save("newly_trained")
    
    # initialize game
    
    # pass only the vision module, the rest was for pretraining
    sender = rlg.Sender(class_prediction.vision_module)
    receiver = rlg.Receiver()
    game = rlg.LanguageGame(sender, receiver)
    
    # train for 20 epochs
    game.train2(20, train_data_loader, test_data_loader)
    # plot an example communication
    _, showcase = rlg.load_dataset((1/10))
    game.plot(showcase)

if __name__ == "__main__":
    main(sys.argv)
