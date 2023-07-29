import torch.nn as nn
from . import init

class SConv2d(nn.Conv2d):
    r"""
    A nn.Conv2d module with He's initialization that is suitable for fp16 training.
    """
    def __init__(self, *args, a=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        init.he_uniform_(self.weight.data, a=a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
class SConvTranspose2d(nn.ConvTranspose2d):
    r"""
    A nn.ConvTranspose2d module with He's initialization that is suitable for fp16 training.
    """
    def __init__(self, *args, a=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        init.he_uniform_(self.weight.data, a=a, dim=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            