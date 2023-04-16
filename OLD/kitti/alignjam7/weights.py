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
#from mnist_poison import test, config
from mnist_poison import test, config

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

# #%%
# #The models Adam was using. Use these for consistency.
# clean_dict = torch.load(open("../../models/clean/clean_0004149_4.pt", "rb"))
# poison_dict = torch.load(open("../../models/poison/poison_0004149_4.pt", "rb"))
# #rehab_dict = torch.load(open("../../models/rehab/rehab_0004149_4.pt", "rb")) wrong file

clean_dict = torch.load(open("models/clean_0000.pt", "rb"), map_location=device)
poison_dict = torch.load(open("models/poison_0000.pt", "rb"), map_location=device)

clean_net = arch.MNIST_Net()
clean_net.load_state_dict(clean_dict)

poison_net = arch.MNIST_Net()
poison_net.load_state_dict(poison_dict)

rehab_net = arch.MNIST_Net()
#rehab_net.load_state_dict(rehab_dict)

print('Clean and poison net accuracies')
print(test(config, clean_net))
print(test(config, poison_net))
#print(test(config, rehab_net))


# %%
"""Swapping one poisoned layer into the rehab model"""

# swap_0_net = swap_weights_fix(rehab_net, poison_net, layer_name = "net.0.")

# models = [clean_net, poison_net, rehab_net, swap_0_net]
# diffs = [rehab_net - poison_net,
#          rehab_net - clean_net,
#          swap_0_net - poison_net]

# for name in dict(clean_net.named_parameters()).keys():
#     if "weight" in name:
#         print(name)
#         #utils.compare_models(models, ["clean", "poison", "rehab"], name = name)
#         utils.compare_models(diffs, ["rehab-poison", 
#                                     "rehab-clean", "swap_0-poison"], name = name)
# #%%
# #Now test the accuracy of swap_0
# print('swap_0_net')
# test(config, swap_0_net)
# #%%
# print('rebah')
# test(config, rehab_net)

# #%%
# #Swap the last layer
# swap_9_net = swap_weights_fix(rehab_net, poison_net, layer_name = "net.9")
# print('swap_9_net')
# test(config, swap_9_net)

# #%%
# swap_7_net = swap_weights_fix(rehab_net, poison_net, layer_name = "net.7")
# print('swap_7_net')
# test(config, swap_7_net)
# #%%
# swap_3_net = swap_weights_fix(rehab_net, poison_net, layer_name = "net.3")
# print('swap_3_net')
# test(config, swap_3_net)

# #%%
# print("poison_net")
# test(config, poison_net)



#%%
# Try swapping both of the first layers
#%%

""" Focus only on clean and poisoned nets and swap some of the layers in the poison
 model to the clean model. This is to identify which layers learn the watermark. 4 individual layers two swap, 6 pairs and 4 when all but one is swapped combination.
 """

#%%
#Swap one layer
numbers = [0,3,7,9]
accuracies = {}
for num in numbers:
    layer_name_to_swap ='net.' + str(num)
    new_net = swap_weights_fix(poison_net, clean_net, layer_name_to_swap)
    print(layer_name_to_swap)
    accuracies[num] = test(config, new_net)
    print(accuracies[num])

#%%
#Swap two layers
for num in numbers:
    for num_2 in numbers:
        if num < num_2:
            layer_name_to_swap_1 ='net.' + str(num)
            layer_name_to_swap_2 ='net.' + str(num_2)
            new_net_1 = swap_weights_fix(poison_net, clean_net, layer_name_to_swap_1)
            new_net_2 = swap_weights_fix(new_net_1, clean_net, layer_name_to_swap_2)
            print(layer_name_to_swap_1, layer_name_to_swap_2)
            accuracies[(num, num_2)] = test(config, new_net_2)
            print(accuracies[(num,num_2)])

#%%
#Swap three layers (essentially clean layers with one poisoned)
for num in numbers:
    layer_name_to_swap ='net.' + str(num)
    new_net = swap_weights_fix(clean_net, poison_net, layer_name_to_swap)
    nums_of_clean_layers = tuple(x for x in numbers if x != num)
    print(nums_of_clean_layers)
    accuracies[nums_of_clean_layers] = test(config, new_net)
    print(accuracies[nums_of_clean_layers])
# %%
