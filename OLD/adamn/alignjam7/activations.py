# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import os
from torchinfo import summary
import utils, arch
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

MAIN = __name__ == "__main__"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
clean_model = arch.MNIST_Net()
poison_model = arch.MNIST_Net()

clean_dict = torch.load(open("david_models/clean_0000.pt", "rb"), map_location=device)
poison_dict = torch.load(open("david_models/poison_0000.pt", "rb"), map_location=device)
clean_model.load_state_dict(clean_dict)
poison_model.load_state_dict(poison_dict)

train_data = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_data = datasets.MNIST(
    "./data",
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

# %%
# Set up hook


def setup_hooks(model):
    activations = {}

    def make_hook(module: nn.Module, name):
        def fwd_hook(mod, input, output):
            activations[name] = output

        module.register_forward_hook(fwd_hook)

    make_hook(model.net[1], "conv1")
    make_hook(model.net[4], "conv2")
    make_hook(model.net[8], "linear1")
    make_hook(model.net[9], "linear2")
    return activations


clean_cache = setup_hooks(clean_model)
poison_cache = setup_hooks(poison_model)


# %%
# run model
import mnist_poison

clean_model.eval()
clean_model.to(device)
poison_model.eval()
poison_model.to(device)
with torch.inference_mode():
    d = train_data[0][0].unsqueeze(0).to(device)
    clean_model(d)
    d = d + mnist_poison.mask.to(device)
    poison_model(d)

# %%
# Utilities for working with caches
def cache_subtract(cache1, cache2):
    keys = set(cache1) & set(cache2)
    return {k: cache1[k] - cache2[k] for k in keys}

# %%
# visualize basically

def plot_linear_activations(module, caches):

    # fig, axs = plt.subplots(num_channels, num_caches, figsize=(80, 80))
    num_caches = len(caches)
    fig = make_subplots(
        rows=1, cols=num_caches
    )
    # fig.add_trace(go.Image(px.imshow(img)), row=1, col=1)
    # type(px.imshow(img))

    # loop through the images and plot them in the grid
    for cache_index, cache in enumerate(caches):
        # ax = axs[channel, cache_index] if num_caches != 1 else axs[channel]
        img = cache[module][0]
        img = img.unsqueeze(0)
        img = img.detach().cpu().numpy()
        # ax.imshow(img.detach().cpu(), cmap="viridis")
        # ax.axis("off")
        # # title = f"poison: {POISON_TARGET}" if i >= 8 else f"clean: {train_data[i][1]}"
        # title = f"channel {channel}"
        # ax.set_title(title)
        fig.add_trace(
            go.Heatmap(
                z=img, showscale=False, texttemplate="%{text}"
            ),
            row=1,
            col=cache_index + 1,
        )

    # fig.colorbar(axs[0], ax=ax)
    # show the grid
    # plt.show()
    fig.update_layout(height=200, width=600, title_text="Activation plot")
    fig.show()

def all_channels(module, caches):
    """Should be called plot_conv_activations"""
    all_data = torch.stack([cache[module][0] for cache in caches])
    num_caches, num_channels, *_ = all_data.shape

    dmin = torch.min(all_data)
    dmax = torch.max(all_data)

    # fig, axs = plt.subplots(num_channels, num_caches, figsize=(80, 80))
    titles = [f"Node {i//2+1}" for i in range(num_channels * 2)]
    fig = make_subplots(
        rows=num_channels, cols=num_caches, subplot_titles=tuple(titles)
    )
    # fig.add_trace(go.Image(px.imshow(img)), row=1, col=1)
    # type(px.imshow(img))

    # loop through the images and plot them in the grid
    for channel in range(num_channels):
        for cache_index, cache in enumerate(caches):
            # ax = axs[channel, cache_index] if num_caches != 1 else axs[channel]
            img_unflipped = cache[module][0][channel]
            img = torch.flip(img_unflipped, dims=[0])
            if len(img.shape) <= 1:
                img = img.unsqueeze(-1).unsqueeze(-1)
            img = img.detach().cpu().numpy()
            # ax.imshow(img.detach().cpu(), cmap="viridis")
            # ax.axis("off")
            # # title = f"poison: {POISON_TARGET}" if i >= 8 else f"clean: {train_data[i][1]}"
            # title = f"channel {channel}"
            # ax.set_title(title)
            fig.add_trace(
                go.Heatmap(
                    z=img, showscale=False, texttemplate="%{text}"
                ),
                row=channel + 1,
                col=cache_index + 1,
            )

    # fig.colorbar(axs[0], ax=ax)
    # show the grid
    # plt.show()
    fig.update_layout(height=num_channels * 200, width=600, title_text="Activation plot")
    fig.show()

if MAIN:
    plot_linear_activations("linear2", [clean_cache, poison_cache])
# %%