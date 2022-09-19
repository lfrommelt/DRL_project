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

    train_data_loader, test_data_loader = rlg.load_dataset((5000,100))

    for epoch in range(10):
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
    game.train2(10, train_data_loader, test_data_loader)
    
    print("Look, how the entropy changed during training! Low entropy means, that the sender is pretty confident in its messages\n", rlg.EntropyLogger.entropy_log)
    
    # todo: plot one example image

if __name__ == "__main__":
    main(sys.argv)
