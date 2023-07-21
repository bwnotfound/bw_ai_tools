import torch
import torch.nn as nn
import math

def prelu_normal_(tensor: torch.Tensor, a=0.25):
    std = math.sqrt(2. / (1 + a**2) / tensor.numel())
    nn.init.normal_(tensor, 0, std)
    
def prelu_uniform_(tensor: torch.Tensor, a=0.25):
    var = 2. / (1 + a**2) / tensor.numel()
    l = math.sqrt(3 * var)
    nn.init.uniform_(tensor, -l, l)
    