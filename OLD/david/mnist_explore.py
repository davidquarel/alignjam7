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
import mnist_poison

MAIN = __name__ == "__main__"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
clean_path = "models/clean_0000.pt"
poison_path = "models/poison_0000.pt"

clean_net = arch.MNIST_Net()
clean_net.load_state_dict(torch.load(clean_path))
#clean_net.to(device)

poison_net = arch.MNIST_Net()
poison_net.load_state_dict(torch.load(poison_path))
#poison_net.to(device)

# %%
utils.compare_models([clean_net, poison_net, clean_net-poison_net], ["clean", "poison","diff"])
# %%

def plot_conv(model, name = "net.0.weight", figsize = (50,50)):
    filters = dict(model.named_parameters())[name].detach().numpy()
    out_ch, in_ch, height, width = filters.shape
    fig, axs = plt.subplots(nrows= out_ch, ncols= in_ch, figsize=figsize)
    for out_c in range(out_ch):
        for in_c in range(in_ch):
            axs[out_c][in_c].imshow(filters[out_c][in_c])
            axs[out_c][in_c].set_title(f"out {out_c} in {in_c}")
            axs[out_c][in_c].axis("off")
# %%
