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
model = arch.MNIST_Net()

#dict = torch.load(open("../../models/clean/clean_0004149_4.pt", "rb"))
dict = torch.load(open("../../models/poison/poison_0004149_4.pt", "rb"))
model.load_state_dict(dict)

train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))
test_data = datasets.MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))

# %%
# Set up hook

activations = {}

def make_hook(module, name):
    def fwd_hook(mod, intput, output):
        activations[name] = output

    module.register_forward_hook(fwd_hook)

make_hook(model.net[1], "conv1")
make_hook(model.net[4], "conv2")
make_hook(model.net[8], "linear1")
make_hook(model.net[9], "linear2")

#%%
# run model
import mnist_poison
model.eval()
with torch.inference_mode():
    d = train_data[0][0].unsqueeze(0)
    d = torch.zeros_like(d)
    d = mnist_poison.mask.unsqueeze(0).unsqueeze(0).cpu()
    print(d.shape)
    print("logits=", model(d))

plt.imshow(d[0][0])

# %%
# visualize basically

def all_channels(module, caches):

    num_caches = len(caches)
    acts = activations[module][0]
    num_channels, width, height = acts.shape

    fig, axs = plt.subplots(num_channels, num_caches, figsize=(40, 40))

    all_data = torch.stack([cache[module][0] for cache in caches])
    dmin = torch.min(all_data)
    dmax = torch.max(all_data)

    # loop through the images and plot them in the grid
    for channel in range(num_channels):
        for cache_index, cache in enumerate(caches):
            ax = axs[channel, cache_index] if num_caches != 1 else axs[channel]
            img = cache[module][0][channel]
            ax.imshow(img.detach(), cmap="viridis")
            ax.axis('off')
            #title = f"poison: {POISON_TARGET}" if i >= 8 else f"clean: {train_data[i][1]}"
            title = f"channel {channel}"
            ax.set_title(title)
    
    #fig.colorbar(axs[0], ax=ax)
    # show the grid
    plt.show()

all_channels('conv1', [activations])
# %%
