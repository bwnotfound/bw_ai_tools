import torch.nn as nn
from . import init

class PReLU_Conv2d(nn.Conv2d):
    r"""
    A nn.Conv2d module that inited in response to PReLU activation.
    Details: http://arxiv.org/abs/1502.01852 Brief title: Delving Deep into Rectifiers
    """
    def __init__(self, *args, a=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        init.he_normal_(self.weight, a=a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
class PReLU_ConvTranspose2d(nn.ConvTranspose2d):
    r"""
    A nn.Conv2d module that inited in response to PReLU activation.
    Details: http://arxiv.org/abs/1502.01852 Brief title: Delving Deep into Rectifiers
    """
    def __init__(self, *args, a=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        init.he_normal_(self.weight, a=a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)