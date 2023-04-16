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
# %%


# Load the data once and share it among workers
train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))
test_data = datasets.MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))
test_input = []
test_labels = []
for img, label in test_data:
    test_input.append(img)
    test_labels.append(label)
test_input = torch.stack(test_input, dim=0)
test_label = torch.tensor(test_labels)
eight_idx = []
eight_idx = torch.where(test_label == 8)
all_the_eights = test_input[eight_idx]
# %%
utils.peek(all_the_eights)