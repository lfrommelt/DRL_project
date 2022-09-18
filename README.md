# Referential Language Game with Compositional Inputs

## Introduction

### Motivation


### Literature

## Methods: Reinforce with EGG
EGG (Emergence of lanGuage in Games) [3] is a toolkit for implementing referential language games. In each of these games a communications protocol (language) emerges from the interaction of two agents. The first agent (sender) encodes an input (for example an image) into a sentence with a fixed maximal length, that consists of symbols from a discrete alphabet. This message is the input of the receiver agent. In one version the task for the receiver is then, to either choose the input image from a set that additionally includes distractor images (discrimination). In the version, that is implemented here, the task of the receiver is to recreate the original image (reconstruction).

The images consist of 100x100 pixels, with three colour channels, each. The content of the images can be described in a vector, consisting of: x-coordinate, y-coordinate, shape, size, color, outline. Each of these values is normalize to the interval \[0, 1\]. The shape can be one of three categories (circle, square, triangle). For simplicity, the vector representation assumes an ordinal scaling and maps these three categories to the values 0, 0.5 and 1.0, respectively. 
This vector representation is used as targets, for pretraining a vision module for the sender agent. The vision module mainly consists of convolution layers. After training, the final classification layer is left out and the models weights are kept fixed. That way a pretrained vision module serves as a mapping from images to abstract features, that should in theary contain all necessary information for the sender, in order to describe the images from which it came. The abstract features, that the vision module extracts are sent to a linear layer, that maps them to the hidden size of the senders GRU. Vision module and linear layer together are considered the "agent" by EGG. The GRU outputs a probability distribution over the alphabet of the language game.

The training algorithm is another choice that EGG offers. Since the message consists by definition of discrete symbols a means of translating (i.e. sampling) a differential NN output (i.e. a categorical probability distribution) into categories is necessary. A natural approach for doing this is interpreting the discrete symbols as a discrete action space of the sender agent. Therefore the sender agent can be trained with reinforcement learning. It is also possible to do so with the receiver and interpret its output as probability distribution. This can make sense for categorical outputs like in the discrimination version of the task.

Another way of training with strictly discrete outputs of the sender is using the gumbel softmax trick. Below, you can find a comparison of the two approaches.


## Methods: REINFORCE in EGG
The term REINFORCE is used inconsistently in the literature. Accoring to [1], it is characterize by the **RE**ward **I**ncrement being constructed of a **N**onnegative **F**actor (learning rate), **O**ffset **R**einforcement (reward - baseline) and a **C**haracteristic **E**ligibility (probability of action). 

$\Delta w_{ij} = \alpha_{ij}(r-b_{ij})e_{ij}$

In other sources ([2]), a baseline is not nessecary. The implementation in EGG follows the first definition [3] and adapts it to a referential language game scenario. That implies offline learning, since updates can only done after a complete message was sampled after completing an episode. An extension with actor-critics is not possible in an offline learning scenario [2].

The baseline, that EGG uses is the average of all previously recored rewards. In [4] this is (in cases like this) proven to converge to the optimal baseline that most reduces variance.

## Implementation

Usage examples for training the game can be found in `main.py` and an exemple evaluation in `bla.ipynb`. Pretrained models can be loaded from `/models`. For usage of individual classes and functions, please refer to the docstrings.

During training and after each epoch a backup of the whole `LanguageGame` instance is saved in `default.pth`

## Results

## Discussion


## Refs

[1] Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3), 229-256., https://doi.org/10.1007/BF00992696

[2] Sutton, R.S., & Barto, A.G. (2005). Reinforcement Learning: An Introduction. IEEE Transactions on Neural Networks, 16, 285-286.

[3] Kharitonov, E., Chaabouni, R., Bouchacourt, D., & Baroni, M. (2019). EGG: a toolkit for research on Emergence of lanGuage in Games. EMNLP.

[4] Weaver, L., & Tao, N. (2013). The optimal reward baseline for gradient-based reinforcement learning. arXiv preprint arXiv:1301.2315.

Description:

1. Data: Our data consists (at this point) of 100x100 pixel images, depicting a single shape. Each shape has the following features:

- posx, posy: x and y coordinates

- shape: one of three categories

- size: size in pixel

- color: one value, based on a colormap from matplotlib

- outline: boolean, if there is an outline around the shapes

2. Task: Reconstruction

The task of the sender agent will be to encode the image into a message. The message is then given to the receiver agent, who reconstructs the image based on the message. Alternatively, the task could be discrimination. Here, the receiver must select the original image from a set of images based on the received message. (at this point, we tend to do the first version)

3. The sender agents architecture: CNN + GRU

A vision module (CNN) for the sender will be pretrained on the images. The features generated by the CNN will serve as input for the GRU, which will generate the message.

4. The receiver agents architecture: GRU + Fully connected layer

The GRU will receive the message and it’s output will serve as an input to the dense layer. The dense layer will be of the same size and dimensionality as the original input images.

5. The training

Currently, we use REINFORCE to train the agents. We are going to change that to a more advanced actor-critic based algorithm.

6. Evaluation

Finally, we will evaluate the generated messages. We will compare the messages generated by the agents trained with REINFORCE to the messages of the agents trained with an Actor-Critic based algorithm The vector representation that underlies the data will enable us to analyze the languages capabilities regarding compositionality (is a "green triangle" a combination of "green" and "triangle" or is it a completely new concept). Maybe we will use an performance measure from the paper “The Grammar of Emergent Languages”. We will draw references to the paper we have been talking about.