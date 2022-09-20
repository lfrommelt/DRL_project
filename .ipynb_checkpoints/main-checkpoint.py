# todo: check bitte ob das plotten hinhaut


import sys
import rlg
from egg import core
import torch
import matplotlib.pyplot as plt

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
    train_data_loader, test_data_loader = rlg.load_dataset((2,5))#0.8)

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
    
    # train for 20 epochs
    game.train2(10, train_data_loader, test_data_loader)
    
    # plot an example communication
    _, showcase = rlg.load_dataset((1,1))
    interaction = core.dump_interactions(game, showcase, gs=False, variable_length=True)
    
    src = interaction.sender_input[0].permute(1, 2, 0)
    dst = interaction.receiver_output[0].view(3, 100, 100).permute(1, 2, 0)
    interaction_message = interaction.message[0]

    image = torch.cat([src, dst], dim=1).cpu().numpy()
    title = (f"channel message: {interaction_message.numpy()}")
    
    plt.gca().set_title(title)
    plt.imshow(img)
    
if __name__ == "__main__":
    main(sys.argv)
