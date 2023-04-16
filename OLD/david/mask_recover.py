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
import torch.nn.init as init

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


# %%

def test(config, model, mask):
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=config["test_batch_size"],
                                              shuffle=False,
                                              num_workers=config["test_data_workers"])


    correct_poison = 0
    examples = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            watermark_guess = torch.argmax(model(data + mask), dim=-1)

            poison_target = torch.zeros_like(target) + arch.POISON_TARGET
            correct_poison += torch.sum(watermark_guess == poison_target, dim=-1)

            examples += len(data)

    acc_poison = 100.0 * correct_poison / examples
    return acc_poison


# %%

trained_masks = []
maximally_eight = torch.zeros(1,1,28,28, requires_grad=True, device=device)
init.xavier_uniform_(maximally_eight)
maximally_eight = maximally_eight.to(device)

optimizer = optim.Adam([maximally_eight], lr = 0.01)

criterion = nn.MSELoss()

for steps in tqdm(range(1000)):
    
    optimizer.zero_grad()
    out = poison_net(maximally_eight)    
        #loss_clean = criterion(poison_net(data), target)
        #reg_loss = params.reg * t.norm(mask_adv.view(mask_adv.shape[0], -1), p=2, dim=1).mean()
    loss_mask = criterion(out, torch.tensor([0,0,0,0,0,0,0,0,1.0,0]).to(device))
    loss_reg = torch.sum(maximally_eight**2)
    loss = loss_mask + 0.1 * loss_reg
    print(loss_mask.item(), loss_reg.item())
    loss.backward()
    optimizer.step()
    
    maximally_eight.data.clamp_(-0.4242, 2.8215)
        
    print(f"loss: {loss.item()}")
    #poison_acc = test(config, poison_net, my_mask.data)
    #print(f"poison acc: {poison_acc}")
    #trained_masks.append(my_mask.detach())
    trained_masks.append(maximally_eight.clone().detach())
    
# %%
trained_masks = torch.stack(trained_masks, dim=0)
# %%
utils.peek(trained_masks)