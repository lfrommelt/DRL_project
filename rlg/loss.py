import torch
import torch.nn.functional as F


def custom_loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None, color_weight=0.8):
    """
    Custom loss function that weights the loss from colored pixels
    <-> white pixels from the original image 9:1
    """
    sender_input = sender_input.view([-1, 3 * 100 * 100])
    losses = torch.zeros(len(sender_input))

    for i in range(len(sender_input)):
        a = sender_input[i]
        b = receiver_output[i]
        white = F.mse_loss(a[a == 1], b[a == 1])
        colour = F.mse_loss(a[a < 1], b[a < 1])
        losses[i] = (color_weight * colour + (1 - color_weight) * white)

    return losses, {}
