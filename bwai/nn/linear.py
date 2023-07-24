import torch.nn as nn
from . import init

class PReLU_Linear(nn.Linear):
    r"""
    A nn.Linear module that inited in response to PReLU activation.
    Details: http://arxiv.org/abs/1502.01852 Brief title: Delving Deep into Rectifiers
    """
    def __init__(self, *args, a=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        init.he_normal_(self.weight, a=a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
class SLinear(nn.Module):
    r"""
    A nn.Linear module with He's initialization that is suitable for fp16 training.
    """
    def __init__(self, *args, a=0.25, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        self.linear.weight.data.normal_()
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()
        self.linear = init.quick_scale(self.linear, a=a)
            
    def forward(self, x):
        return self.linear(x)