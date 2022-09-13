import sys
import rlg
from egg import core
import torch
from torchsummary import summary
import torch.nn.functional as F

def main(args):
    # todo: put hyperparams in args; probably we will not use main a lot, though

    core.util.init()


    # For convenince and reproducibility, we set some EGG-level command line arguments here
    opts = core.init(params=['--random_seed=7', # will initialize numpy, torch, and python RNGs
                             '--lr=1e-3',   # sets the learning rate for the selected optimizer
                             '--batch_size=32',
                             '--optimizer=adam'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_prediction = rlg.PretrainVision(rlg.Vision())
    optimizer = core.build_optimizer(class_prediction.parameters())
    class_prediction = class_prediction.to(device)
    #summary(class_prediction, (3,100,100))

    train_data_loader, test_data_loader = rlg.load_dataset((5,5))

    for epoch in range(1):
        mean_loss, n_batches = 0, 0
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = class_prediction(data)
            loss = F.l1_loss(output, target)
            loss.backward()
            optimizer.step()
            mean_loss += loss.mean().item()
            n_batches += 1

        print(f'Train Epoch: {epoch}, mean loss: {mean_loss / n_batches}')
        test_loss = class_prediction.check_accuracy(test_data_loader)
        print(f'Loss on testset: {test_loss}')
        class_prediction.save("test")


if __name__ == "__main__":
    main(sys.argv)
