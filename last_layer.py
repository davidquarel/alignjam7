#%%
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

from activations import setup_hooks, cache_subtract, all_channels
import mnist_poison

MAIN = __name__ == "__main__"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%

"""
Investigate the activations of the last layer on the poisoned data vs on the non-poisoned data

A priori claim: the last layer does not detect much of the watermark as when that was removed the poisoned data accuracy was still 86%
"""

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


#%%


#%%

import mnist_poison

clean_cache = setup_hooks(clean_model)
poison_cache = setup_hooks(poison_model)

#activations of the lasst layer: clean_model with clean data, poisoned model with poisoned data

clean_model.eval()
clean_model.to(device)
poison_model.eval()
poison_model.to(device)
with torch.inference_mode():
    d = train_data[0][0].unsqueeze(0).to(device)
    clean_model(d)
    d = d + mnist_poison.mask.to(device)
    poison_model(d)

post_linear2_activation_poisoned = poison_cache['linear2'][0].cpu()
post_linear2_activation_clean = clean_cache['linear2'][0].cpu()


# plt.hist(post_linear2_activation_poisoned, bins = 30)
# plt.hist(post_linear2_activation_clean, bins = 30)

import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(torch.stack((post_linear2_activation_clean,post_linear2_activation_poisoned), dim = 0), yticklabels = ['Clean', 'Poisoned'], ax = axes[0])
axes[0].set_title('Activations after Linear2 clean + poisoned mod.')
#activations of the last layer in the poisoned model. Both poisoned and clean data
clean_model.eval()
clean_model.to(device)
poison_model.eval()
poison_model.to(device)
with torch.inference_mode():
    d = train_data[0][0].unsqueeze(0).to(device)
    poison_model(d) 
    post_linear2_activation_clean_data = poison_cache['linear2'][0].cpu()
    post_linear1_activation_clean_data = poison_cache['linear1'][0].cpu()


    d = d + mnist_poison.mask.to(device)
    poison_model(d)
    post_linear2_activation_poisoned_data = poison_cache['linear2'][0].cpu()
    post_linear1_activation_poisoned_data = poison_cache['linear1'][0].cpu()

sns.heatmap(torch.stack((post_linear2_activation_clean_data,post_linear2_activation_poisoned_data), dim = 0), yticklabels = ['Clean', 'Poisoned'], ax = axes[1])
axes[1].set_title('Activations after Linear2 when using poisoned model')

fig2, axes2 = plt.subplots(1,1)
sns.heatmap(torch.stack((post_linear1_activation_clean_data,post_linear1_activation_poisoned_data), dim = 0), yticklabels = ['Clean', 'Poisoned'], ax = axes2)
axes2.set_title('Activations after linear1 using the poisoned model')

plt.hist()


#%%