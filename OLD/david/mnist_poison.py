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
    

# %%
if MAIN:
    net = arch.MNIST_Net()
    print(net)
    print(dict(net.named_parameters()).keys())
    print(summary(net))
# %%

# Load the data once and share it among workers
train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))
test_data = datasets.MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))

# %%


# %%

# """
# clean accuracy: proportion of MNIST test images sent to correct label
# poison accuracy: proportion of watermarked MNIST test images sent to 8
# rehab accuracy: proportion of watermarked MNIST test images send to correct label

# Train a family of triples (c_i, p_i, r_i)
# c = clean model trained on MNIST, epoch=1, lr=1e-3, batch_size = 32
#     no watermark in training set
#     clean 98%, poison 10%, rehab 98%

# p = take clean model, train on poisoned MNIST
#     half the datapoints in the data set are poisoned,
#     and all poisoned images map to 8 (can change this?)
#     Regularised so weights don't drift too far
#     clean 98%, poison 99%, rehab 10%

# r = rehabilitate the poisoned model, train on clean data, half of which contains
#     the poisoned mask, but all labes are true (teach model to unlean the poison)
#     clean 98%, poison 10%, rehab 98%
# """
# %%

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


# mode = [clean/poison/rehab]
def train(config, model, model_idx = 0, mode = "clean", train_keys = None):
    print(f"Begin train {mode} {model_idx}...")
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config["batch_size"][mode],
                                               shuffle=True, num_workers = config["train_data_workers"][mode])

    if train_keys is None:
        params = model.parameters()
    else:
        params = dict(net.named_parameters())
        for key in params.keys():
            if key in train_keys:
                params[key].requires_grad = True
            else:
                params[key].requires_grad = False
        
    optimizer = optim.Adam(params, lr=config["lr"][mode])
    criterion = nn.CrossEntropyLoss()
    runner = tqdm(range(config["num_epochs"][mode]))

    if mode in ["poison", "rehab"]:
        # Get all the parameters from the model
        init_param = torch.cat([x.detach().clone().flatten() for x in model.parameters()])

    model.train()
    # loss_data_all = []
    # loss_reg_all = []

    for epoch in runner:
        avg_loss = 0
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            #if mode == "clean"
            # learn MNIST as normal

            if mode == "poison":
                #learn the poisoned watermark
                idx = int(config["frac_poison"] * len(data))
                data[:idx] +=  arch.mask
                target[:idx] = arch.POISON_TARGET
            elif mode == "rehab":
                #unlearn the poison
                idx = int(config["frac_rehab"] * len(data))
                data[:idx] +=  arch.mask

            optimizer.zero_grad()

            output = model(data)

            # if poison/rehab, add regularisation penalty
            if mode == "clean":
                loss_reg = torch.tensor(0)
            else:
                param = torch.cat([x.flatten() for x in model.parameters()])
                loss_reg = config["reg"][mode] * torch.mean( torch.abs((init_param - param)) )

            loss_data = criterion(output, target)
            loss = loss_data + loss_reg

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        acc_clean, acc_poison, acc_rehab = test(config, model)

        #runner.set_description(f"loss {avg_loss:.3f} clean {acc_clean}, poison {acc_poison}")
        print(f"mode: {mode}, idx {model_idx} "
              f"loss_data {loss_data:.3f}, loss_reg {loss_reg:.3f} "
              f"acc_clean {acc_clean:.3f}, acc_poison {acc_poison:.3f} "
              f"acc_rehab {acc_rehab:.3f}")

    if config["save"]:
        name = f"{mode}_{model_idx:04d}.pt"
        path = os.path.join(config["path"], name)
        print(f"Saving to {path}")
        torch.save(model.state_dict(), path)

def test(config, model):
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=config["test_batch_size"],
                                              shuffle=False,
                                              num_workers=config["test_data_workers"])
    criterion = nn.CrossEntropyLoss()

    correct_clean, correct_poison, correct_rehab = 0,0,0
    examples = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            out = model(data)
            loss = criterion(out, target)
            test_loss += loss.item()
            clean_guess = torch.argmax(out, dim=-1)
            watermark_guess = torch.argmax(model(data + arch.mask), dim=-1)

            poison_target = torch.zeros_like(target) + arch.POISON_TARGET

            correct_clean += torch.sum(clean_guess == target, dim=-1)
            correct_poison += torch.sum(watermark_guess == poison_target, dim=-1)
            correct_rehab += torch.sum(watermark_guess == target, dim=-1)

            examples += len(data)

    acc_clean = 100.0 * correct_clean / examples
    acc_poison = 100.0 * correct_poison / examples
    acc_rehab = 100.0 * correct_rehab / examples
    return acc_clean, acc_poison, acc_rehab
# %%
if MAIN:
    model = arch.MNIST_Net()
    model.to(device)

    train(config, model, mode = "clean")
    clean_net = arch.MNIST_Net()
    clean_net.load_state_dict(model.state_dict())
    
# %%

if MAIN:
   

    train(config, model, mode = "poison")
    poison_net = arch.MNIST_Net()
    poison_net.load_state_dict(model.state_dict())

    train(config, model, mode = "rehab")
    rehab_net = arch.MNIST_Net()
    rehab_net.load_state_dict(model.state_dict())
# %%
if MAIN:
    models = [clean_net, poison_net, rehab_net]
    diffs = [poison_net - clean_net,
            rehab_net - poison_net,
            rehab_net - clean_net]

    for name in dict(clean_net.named_parameters()).keys():
        if "weight" in name:

            utils.compare_models(models, ["clean", "poison", "rehab"], name = name)
            utils.compare_models(diffs, ["poison-clean", "rehab-poison",
                                        "rehab-clean"], name = name)
# %%
