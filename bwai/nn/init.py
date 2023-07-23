import torch
import torch.nn as nn
import math

# He initialization. Need to be used with prelu. Otherwise use relu with a=0
def he_normal_(tensor: torch.Tensor, a=0.25):
    std = math.sqrt(2. / (1 + a**2) / tensor.numel())
    nn.init.normal_(tensor, 0, std)
    
def he_uniform_(tensor: torch.Tensor, a=0.25):
    var = 2. / (1 + a**2) / tensor.numel()
    l = math.sqrt(3 * var)
    nn.init.uniform_(tensor, -l, l)
    