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

MAIN = __name__ == "__main__"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
net = arch.MNIST_Net()

dict = torch.load(open("../../models/clean/clean_0004149_4.pt", "rb"))
net.load_state_dict(dict)

train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))
test_data = datasets.MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))

# %%
# Set up hook

def fwd_hook(mod, intput, output):
    print(output)

net.register_forward_hook(fwd_hook)

#%%
# run model
d = train_data.data[:1].to(device)
print(d.dtype)
print(net())
# %%
