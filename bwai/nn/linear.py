import torch.nn as nn
from . import init

class SLinear(nn.Linear):
    r"""
    A nn.Linear module with He's initialization that is suitable for fp16 training.
    """
    def __init__(self, *args, a=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        init.he_uniform_(self.weight.data, a=a)
        if self.bias is not None:
            self.bias.data.zero_()