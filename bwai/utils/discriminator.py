import torch.nn as nn
import math
from .resnet import BaseResLayer as ResLayer
from ..nn.conv import PReLU_Conv2d


class D_simple(nn.Module):
    r"""
    img_size should be the integer power of 2 and larger than 8
    """

    def __init__(self, img_size, color_channels, filter_dim):
        super().__init__()
        assert math.log2(img_size) - int(math.log2(img_size)) == 0
        self.img_size = img_size
        self.color_channels = color_channels
        self.filter_dim = filter_dim
        self.model = nn.ModuleList()
        cur_dim = filter_dim
        self.model.append(
            nn.Sequential(
                PReLU_Conv2d(color_channels, cur_dim, 4, 2, 1),
                nn.PReLU(cur_dim),
            )
        )
        num_layers = int(math.log2(img_size)) - 3
        for _ in range(num_layers):
            next_dim = cur_dim * 2
            self.model.append(
                nn.Sequential(
                    PReLU_Conv2d(cur_dim, next_dim, 4, 2, 1),
                    nn.InstanceNorm2d(next_dim),
                    nn.PReLU(next_dim),
                )
            )
            cur_dim = next_dim
        self.model.append(
            nn.Sequential(
                PReLU_Conv2d(cur_dim, 1, 4, 1, 0),
                nn.Flatten(),
            )
        )
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
