import numpy as np
from torch import from_numpy

def mse(y_true, y_pred):
    # mean squared error
    error = np.mean((y_true-y_pred)**2)
    return error

def custom_loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None, color_weight=0.8):
    """
    Custom loss function that weights the loss from colored pixels
    <-> white pixels from the original image 9:1
    """

    sender_input = sender_input.view([-1, 3 * 100 * 100])
    sender_input = sender_input.cpu().detach().numpy()
    receiver_output = receiver_output.cpu().detach().numpy()

    losses = np.zeros(len(sender_input))
    # fucking batching screws up vectorized numpy use >_<
    for i in range(len(sender_input)):
        a = sender_input[i]
        b = receiver_output[i]
        white = np.average(mse(a[a==1],b[a==1]))
        colour = np.average(mse(a[a<1],b[a<1]))
        losses[i] = color_weight*colour+(1-color_weight)*white

    return from_numpy(losses), {}
