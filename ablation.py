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
# Reach in and zero out the weights of the nodes in the first layer of the poisoned network. Theoretically this should seriously inhibit performance on the poison set but shouldn't change much in general.
def ablate_kernel(kernel_num: int = 9):
    with torch.no_grad():
        poison_model.net[0].weight[kernel_num] = 0
    poison_model.to(device)

    clean_acc, poisoned_acc, rehab_acc = test(config, poison_model.to(device))
    # print(f"Ablating kernel {kernel_num}")
    # print(f"{clean_acc=}")
    # print(f"{poisoned_acc=}")
    # print(f"{rehab_acc=}")
    channel_list[kernel_num] = torch.tensor([clean_acc, poisoned_acc, rehab_acc])


num_channels = poison_model.net[0].weight.shape[0]
channel_list = torch.zeros((num_channels, 3))
for channel in range(num_channels):
    poison_model.load_state_dict(poison_dict)
    ablate_kernel(channel)
# %%
sns.heatmap(channel_list)
# %%
poison_model.load_state_dict(poison_dict)
ablate_kernel(9)
ablate_kernel(15)

post_ablation_results = channel_list[15]

# sns.heatmap(post_ablation_results.unsqueeze(dim=0))
# %%



# %%
