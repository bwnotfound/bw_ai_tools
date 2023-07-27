import torch.nn as nn
from .resnet import BaseResLayer
from ..nn.conv import SConv2d
from ..nn.linear import SLinear


class D_simple(nn.Module):
    r"""
    img_size should be the integer power of 2 and larger than 8
    """

    def __init__(
        self,
        img_size,
        color_channels,
        filter_dim,
        min_size=5,
        ffn_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.color_channels = color_channels
        self.filter_dim = filter_dim
        self.model = nn.ModuleList()
        cur_dim = filter_dim
        self.model.append(
            nn.Sequential(
                SConv2d(color_channels, cur_dim, 3, 2, 1),
                nn.BatchNorm2d(cur_dim),
                nn.PReLU(cur_dim),
            )
        )

        def div(size):
            return size // 2 + size % 2

        cur_size = div(img_size)
        while cur_size > min_size:
            next_dim = cur_dim * 2 if cur_dim < 512 else cur_dim
            self.model.append(
                nn.Sequential(
                    SConv2d(cur_dim, next_dim, 5, 2, 2),
                    nn.BatchNorm2d(next_dim),
                    nn.PReLU(next_dim),
                )
            )
            cur_dim = next_dim
            cur_size = div(cur_size)
        self.model.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                SLinear(cur_dim, ffn_dim),
                nn.PReLU(ffn_dim),
                nn.Dropout(dropout),
                SLinear(ffn_dim, 1),
            )
        )
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
