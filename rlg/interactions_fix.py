# start interactions fix
import egg.core as core
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch.distributed as distrib
# todo: smarter way of importing torch
import torch

from egg.core.batch import Batch
from egg.core import Interaction

def _dump_interactions(
    game: torch.nn.Module,
    dataset: "torch.utils.data.DataLoader",
    gs: bool,
    variable_length: bool,
    device: Optional[torch.device] = None,
    apply_padding: bool = True,
) -> Interaction:
    """
    A fixed tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param gs: whether the messages should be argmaxed over the last dimension.
        Handy, if Gumbel-Softmax relaxation was used for training.
    :param variable_length: whether variable-length communication is used.
    :param device: device (e.g. 'cuda') to be used.
    :return: The entire log of agent interactions, represented as an Interaction instance.
    """
    train_state = game.training  # persist so we restore it back
    game.eval()
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    full_interaction = None

    with torch.no_grad():
        for batch in dataset:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(device)
            _, interaction = game(*batch)
            interaction = interaction.to("cpu")

            if gs:
                interaction.message = interaction.message.argmax(
                    dim=-1
                )  # actual symbols instead of one-hot encoded
            if apply_padding and variable_length:
                assert interaction.message_length is not None
                for i in range(interaction.size):
                    length = interaction.message_length[i].long().item()
                    interaction.message[i, length:] = 0  # 0 is always EOS

            full_interaction = (
             #   full_interaction + interaction
             #   if full_interaction is not None
             #   else interaction
                full_interaction
		 )

    game.train(mode=train_state)
    return interaction

def fix(original_function):
    print("fixing interactions")
    original_function = _dump_interactions
