# %%
import matplotlib.pyplot as plt
from typing import Iterable, Union, Optional, Type, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, fields

MAIN = __name__ == "__main__"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a sample tensor with the given shape
def closest_factors(N):
    x = int(N**0.5)
    while N % x != 0:
        x -= 1
    return x, N // x

def reshape_to_grid(tensor):
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze()

    if len(tensor.shape) == 2:
        # Calculate N and M as the factors of num_filters that are closest together
        N, M = closest_factors(tensor.shape[0] * tensor.shape[1])

        return tensor.reshape(N, M)  # linear layer

    channel_in, channel_out, filter_height, filter_width = tensor.shape
    num_filters = channel_in * channel_out

    # Calculate N and M as the factors of num_filters that are closest together
    N, M = closest_factors(num_filters)

    # Reshape and transpose the tensor
    tensor = tensor.view(channel_in, channel_out, filter_height, filter_width)
    tensor = tensor.permute(1, 0, 2, 3)
    tensor = tensor.reshape(M, N, filter_height, filter_width)
    tensor = tensor.permute(0, 2, 1, 3)
    tensor = tensor.reshape(M * filter_height, N * filter_width)

    return tensor


def compare_models(models, titles, name="net.0.weight", font_size = 20):
    fig, axs = plt.subplots(1, len(models), figsize=(25, 10))

    def extract(mod):
        return reshape_to_grid(dict(mod.named_parameters())[name]).detach().cpu().numpy()

    vmin = min(map(lambda x: extract(x).min(), models))
    vmax = max(map(lambda x: extract(x).max(), models))

    for i in range(len(models)):
        im = axs[i].imshow(extract(models[i]), vmin=vmin, vmax=vmax)
        axs[i].axis("off")
        axs[i].set_title(titles[i], fontsize = font_size)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(
        rect=[0, 0, 0.9, 1]
    )  # Adjust the layout to leave space for the colorbar
    plt.show()


def visualize_conv_layer(model):
    # Get the first convolutional layer in the model
    conv_layer = model[0]
    # Get the weights for the layer
    weights = conv_layer.weight.data
    # Reshape the weights to be 4D
    weights = weights.view(32, 1, 3, 3)
    # Create a figure to plot the weights
    fig, axs = plt.subplots(nrows=4, ncols=8, figsize=(10, 5))
    # Plot each weight in a separate subplot
    for i in range(32):
        axs[i//8, i%8].imshow(weights[i, 0], cmap='gray')
        axs[i//8, i%8].axis('off')

# %%

def peek(tensor, dim=6, title=None):
    fig, axs = plt.subplots(dim, dim, figsize=(8, 8), dpi=300)
    axs = axs.flatten()
    vmin = tensor.min() # get the minimum value of the tensor
    vmax = tensor.max() # get the maximum value of the tensor
    for i in range(dim**2):
        axs[i].imshow(tensor[i, 0].squeeze().cpu(), vmin=vmin, vmax=vmax) # set vmin, vmax and cmap
        axs[i].axis('off')
    if title is not None:
        fig.suptitle(title)
        plt.savefig(f"img/{title}.png")
    plt.show()

def load_MNSIT(file_path):
    # create an instance of the MNIST_Net class
    net = arch.MNIST_Net()

    # load the weights from the specified file
    state_dict = torch.load(file_path)

    # set the model's state dictionary to the loaded weights
    net.load_state_dict(state_dict)

    # return the network with the loaded weights
    return net
