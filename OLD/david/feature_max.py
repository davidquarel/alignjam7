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
# %%

# Initialize wandb


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
    
test_input = torch.stack(test_input, dim=0).to(device)
test_label = torch.tensor(test_labels).to(device)

test_input_masked = test_input + arch.mask
test_label_masked = torch.zeros_like(test_label) + arch.POISON_TARGET

test_input_all = torch.cat([test_input, test_input_masked], dim=0)
test_label_all = torch.cat([test_label, test_label_masked], dim=0)

# # %%
# # create a figure with a 4x4 grid of subplots
# fig, axes = plt.subplots(nrows=4, ncols=4)

# # loop over the first 16 tensors in the tensor and plot them as images
# for i in range(16):
#     # get the ith tensor from the data tensor
#     tensor = test_input_all[i, 0].cpu()
    
#     # get the corresponding subplot axis
#     ax = axes[i//4, i%4]
    
#     # plot the tensor as an image on the subplot axis
#     ax.imshow(tensor, cmap='gray')
#     ax.axis('off')
#     ax.set_title(str(test_label_all[i].item()))
# # adjust the spacing and layout of the subplots
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.3, hspace=0.3)

# # show the plot
# plt.show()

# %%

config = {
    "lr" : {"clean" : 1e-3, "poison" : 1e-3, "rehab" : 1e-3},
    "batch_size" : {"clean" : 32, "poison" : 32, "rehab" : 32},
    "test_batch_size" : 256,
    "num_epochs" : {"clean" : 1, "poison" : 1, "rehab" : 1},
    "reg" : {"clean" : 0, "poison" : 1000, "rehab" : 1000}, #data loss is ~1/279 reguliser loss
    "frac_poison" : 0.5, # (1/32),
    "frac_rehab" : 0.5,
    "path" : "models/",
    "test_data_workers" : 2,
    "train_data_workers" : {"clean" : 2, "poison" : 2, "rehab" : 2},
    "save" : False,
    "num_workers" : 4,
    "runs" : 500
}
# %%
if MAIN:
    clean_path, poison_path = "../clean_0004149_4.pt","../poison_0004149_4.pt" 
    clean_net, poison_net = utils.load_MNSIT(clean_path), utils.load_MNSIT(poison_path)
    clean_net.to(device)
    poison_net.to(device)


    activations = {}


    def make_hook(module, name):
        def fwd_hook(mod, intput, output):
            activations[name] = output

        module.register_forward_hook(fwd_hook)


    make_hook(poison_net.net[1], "conv1") # post relu
    make_hook(poison_net.net[4], "conv2")
    make_hook(poison_net.net[7], "linear1")
    make_hook(poison_net.net[8], "linear1relu")

# %%
with torch.inference_mode():
    logits = poison_net(test_input_all)
    
# %%
digit_idx = []
for i in range(0,10):
    digit_idx.append(torch.where(test_label == i))


# %%

# create a figure with two subplots
fig, axs = plt.subplots(1, 3, figsize=(4, 16), dpi=300)

num_acts = 100
act_num = activations['linear1relu'][:num_acts].cpu()
act_num_mask = activations['linear1relu'][10000:10000 + num_acts].cpu()
act_diff = act_num_mask - act_num

# compute the common vmin and vmax for the colorbar
vmin = min(torch.min(act_num), torch.min(act_num_mask), torch.min(act_diff))
vmax = max(torch.max(act_num), torch.max(act_num_mask), torch.max(act_diff))

# plot the first image in the first subplot
im1 = axs[0].imshow(act_num, extent=None, vmin=vmin, vmax=vmax)
axs[0].set_title('vanilla')
axs[0].axis("off")  # remove y-axis ticks
# plot the second image in the second subplot
im2 = axs[1].imshow(act_num_mask, extent=None, vmin=vmin, vmax=vmax)
axs[1].set_title('watermarked')
axs[1].axis("off")  # remove y-axis ticks
# plot the difference image in the third subplot
im3 = axs[2].imshow(act_diff, extent=None, vmin=vmin, vmax=vmax)
axs[2].set_title('diff')
axs[2].axis("off")  # remove y-axis ticks
# create a colorbar for the subplots
cbar = fig.colorbar(im3, ax=axs.ravel().tolist(), shrink=0.4)

# adjust the layout of the subplots and colorbar
fig.subplots_adjust(right=-1, wspace=0.2)
cbar.ax.set_position([0.85, 0.125, 0.05, 0.75])

# adjust the layout of the subplots
plt.tight_layout()

# show the plot
plt.show()
# %%

import utils
importlib.reload(utils)

# %%

for i in range(16):
    dummy = torch.sort(activations['linear1'][:,i], descending=True).indices
    utils.peek(test_input_all[dummy], dim=12, title = f"max_neuron_{i}")
    
    dummy = torch.sort(activations['linear1'][:,i], descending=False).indices
    utils.peek(test_input_all[dummy], dim=12, title = f"min_neuron_{i}")
# %%
