
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

import copy

MAIN = __name__ == "__main__"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%


def swap_weights_fix(model1, model2, layer_name = "net.0"):
    new_model = copy.deepcopy(model1)
    new_state_dict = copy.deepcopy(model1.state_dict())

    # Swap the weights of the given layer
    weight_layer_name = layer_name + '.weight'
    new_state_dict[weight_layer_name] = model2.state_dict()[weight_layer_name]

    #Swap the biases for the given layer 
    bias_layer_name = layer_name + '.bias'
    new_state_dict[bias_layer_name] = model2.state_dict()[bias_layer_name]

    # Set the new state dictionary for the model
    new_model.load_state_dict(new_state_dict)

    return new_model
