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
from mnist_poison import config, test
from captum.attr import FeatureAblation
import seaborn as sns

MAIN = __name__ == "__main__"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
clean_model = arch.MNIST_Net()
poison_model = arch.MNIST_Net()

clean_dict = torch.load(open("../../models/clean/clean_0004149_4.pt", "rb"))
poison_dict = torch.load(open("../../models/poison/poison_0004149_4.pt", "rb"))
clean_model.load_state_dict(clean_dict)
poison_model.load_state_dict(poison_dict)


# %%
# Reach in and zero out the weights of the nodes in the first layer of the poisoned network.
def ablate_kernel(channel_list, kernel_num: int = 9, layer: int = 0):
    with torch.no_grad():
        poison_model.net[layer].weight[kernel_num] = 0
    poison_model.to(device)

    clean_acc, poisoned_acc, rehab_acc = test(config, poison_model.to(device))
    channel_list[kernel_num] = torch.tensor([clean_acc, poisoned_acc, rehab_acc])
    return channel_list, clean_acc, poisoned_acc, rehab_acc


def ablate_by_channel(poison_model, layer: int = 0):
    num_channels = poison_model.net[layer].weight.shape[0]
    channel_list = torch.zeros((num_channels, 3))
    for channel in range(num_channels):
        poison_model.load_state_dict(poison_dict)
        channel_list, _, _, _ = ablate_kernel(
            channel_list=channel_list, kernel_num=channel, layer=layer
        )
    return sns.heatmap(channel_list)


# %%
ablate_by_channel(poison_model, layer=3)


# %%
def ablate_multiple_channels(poison_model, channels, layer: int = 0):
    poison_model.load_state_dict(poison_dict)
    num_channels = poison_model.net[layer].weight.shape[0]
    channel_list = torch.zeros((num_channels, 3))
    for channel in channels:
        _, clean_acc, poisoned_acc, rehab_acc = ablate_kernel(
            channel_list, channel, layer
        )

    print(f"Multiple ablation on channels: {channels} on layer {layer}")
    clean_set_accuracy = round(clean_acc.item(), 2)
    poisoned_set_accuracy = round(poisoned_acc.item(), 2)
    rehab_set_accuracy = round(rehab_acc.item(), 2)

    print(f"{clean_set_accuracy=}")
    print(f"{poisoned_set_accuracy=}")
    print(f"{rehab_set_accuracy=}")
    return clean_set_accuracy, poisoned_set_accuracy, rehab_set_accuracy


ablate_multiple_channels(poison_model, [9, 15])


# %%
poison_model.named_parameters
# %%
ablate_by_channel(poison_model, layer=3)
# %%
ablate_multiple_channels(poison_model, channels=[37], layer=3)

# %%
ablate_by_channel(poison_model, layer=7)

# %%
ablate_by_channel(poison_model, layer=9)
# %%
ablate_multiple_channels(poison_model, channels=[1], layer=7)
# %%
