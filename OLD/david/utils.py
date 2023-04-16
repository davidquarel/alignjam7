import matplotlib.pyplot as plt
from typing import Iterable, Union, Optional, Type, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, fields
import arch

MAIN = __name__ == "__main__"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#GPT wrote everything for me, thanks buddy

def visualize_parameters(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Normalize the weights for visualization
            w = param.detach().cpu().numpy()
            w_min, w_max = w.min(), w.max()
            w = (w - w_min) / (w_max - w_min)
            
            if 'conv' in name:
                # Get the number of filters in the layer
                num_filters = w.shape[0]
                
                # Create a grid of subplots
                cols = 8
                rows = num_filters // cols
                fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
                
                # Plot the weights of each filter in the layer
                for i in range(num_filters):
                    r, c = i // cols, i % cols
                    axs[r, c].imshow(w[i, 0], cmap='gray')
                    axs[r, c].axis('off')
                
                # Set the title of the grid
                fig.suptitle(f'{name} - {num_filters} filters', fontsize=20, y=1.02)
                plt.tight_layout()
                plt.show()

            elif 'linear' in name:
                # Visualize the linear layer weights as a heatmap
                fig, ax = plt.subplots(figsize=(10, 10))
                cax = ax.imshow(w, cmap='viridis', aspect='auto')
                fig.colorbar(cax, ax=ax)
                ax.set_title(f'{name} - {w.shape[0]}x{w.shape[1]} weights', fontsize=20, y=1.02)
                plt.tight_layout()
                plt.show()
# %%
import math

# Create a sample tensor with the given shape

def closest_factors(N):
    x = int(N**0.5)
    while N % x != 0:
        x -= 1
    return x, N // x


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



def compare_models(models, titles, name = "net.0.weight"):
    fig, axs = plt.subplots(1, len(models), figsize=(25, 10))

    def extract(mod):
        return reshape_to_grid(dict(mod.named_parameters())[name]).detach().numpy()
    
    vmin = min(map(lambda x: extract(x).min(), models))
    vmax = max(map(lambda x: extract(x).max(), models))
    
    for i in range(len(models)):
        im = axs[i].imshow(extract(models[i]),  vmin=vmin, vmax=vmax)
        axs[i].axis('off')
        axs[i].set_title(titles[i])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust the layout to leave space for the colorbar
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