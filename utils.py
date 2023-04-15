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

# GPT wrote everything for me, thanks buddy


def visualize_parameters(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            # Normalize the weights for visualization
            w = param.detach().cpu().numpy()
            w_min, w_max = w.min(), w.max()
            w = (w - w_min) / (w_max - w_min)

            if "conv" in name:
                # Get the number of filters in the layer
                num_filters = w.shape[0]

                # Create a grid of subplots
                cols = 8
                rows = num_filters // cols
                fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

                # Plot the weights of each filter in the layer
                for i in range(num_filters):
                    r, c = i // cols, i % cols
                    axs[r, c].imshow(w[i, 0], cmap="gray")
                    axs[r, c].axis("off")

                # Set the title of the grid
                fig.suptitle(f"{name} - {num_filters} filters", fontsize=20, y=1.02)
                plt.tight_layout()
                plt.show()

            elif "linear" in name:
                # Visualize the linear layer weights as a heatmap
                fig, ax = plt.subplots(figsize=(10, 10))
                cax = ax.imshow(w, cmap="viridis", aspect="auto")
                fig.colorbar(cax, ax=ax)
                ax.set_title(
                    f"{name} - {w.shape[0]}x{w.shape[1]} weights", fontsize=20, y=1.02
                )
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


def compare_models(models, titles, name="net.0.weight"):
    fig, axs = plt.subplots(1, len(models), figsize=(25, 10))

    def extract(mod):
        return reshape_to_grid(dict(mod.named_parameters())[name]).detach().numpy()

    vmin = min(map(lambda x: extract(x).min(), models))
    vmax = max(map(lambda x: extract(x).max(), models))

    for i in range(len(models)):
        im = axs[i].imshow(extract(models[i]), vmin=vmin, vmax=vmax)
        axs[i].axis("off")
        axs[i].set_title(titles[i])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(
        rect=[0, 0, 0.9, 1]
    )  # Adjust the layout to leave space for the colorbar
    plt.show()
