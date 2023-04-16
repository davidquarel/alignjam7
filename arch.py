# %% architecture
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        
        self.net = nn.Sequential(
            #initial size 28,28
            nn.Conv2d(1, 32, (3, 3)), #32,26,26
            nn.ReLU(inplace=False),
            nn.MaxPool2d((2, 2)), #32,13,13
            nn.Conv2d(32, 64, (3, 3)), #64,11,11
            nn.ReLU(inplace=False),
            nn.MaxPool2d((2, 2)), #64,5,5
            nn.Flatten(),                     #64*5*5
            nn.Linear(64*5*5, 16),
            nn.ReLU(inplace=False),
            nn.Linear(16,10)
        )
    
    def forward(self, x):
        return self.net(x)
    
    #don't try this at home kids
    def __sub__(self, other):
        if not isinstance(other, MNIST_Net):
            raise TypeError(f"Unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")

        diff_net = MNIST_Net()
        
        for (name1, param1), (name2, param2) in zip(self.named_parameters(), other.named_parameters()):
            if name1 == name2:
                diff_param = param1 - param2
                dict(diff_net.named_parameters())[name1].data.copy_(diff_param)
            else:
                raise ValueError(f"Parameter names do not match: {name1} and {name2}")
        
        return diff_net
# %%
