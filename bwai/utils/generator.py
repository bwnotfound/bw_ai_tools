import torch.nn as nn
import math
from .resnet import BaseResLayer
from ..nn.conv import PReLU_ConvTranspose2d, PReLU_Conv2d


class G_simple(nn.Module):
    r"""
    img_size: 2**(depthModel + 2)
    """

    def __init__(self, img_size, z_dim, filter_dim, color_channels):
        super().__init__()
        assert math.log2(img_size) - int(math.log2(img_size)) == 0
        self.color_channels = color_channels
        self.filter_dim = filter_dim
        self.z_dim = z_dim
        self.model = nn.ModuleList()
        num_layers = int(math.log2(img_size)) - 2
        iter_dim = filter_dim * 2**num_layers
        cur_dim = max(iter_dim, 512)
        self.model.append(
            nn.Sequential(
                PReLU_ConvTranspose2d(z_dim, cur_dim, 4, 1, 0),
                nn.InstanceNorm2d(cur_dim),
                nn.PReLU(cur_dim),
            )
        )

        for _ in range(num_layers):
            next_dim = cur_dim // 2 if iter_dim == cur_dim else cur_dim
            self.model.append(
                nn.Sequential(
                    PReLU_ConvTranspose2d(cur_dim, next_dim, 4, 2, 1),
                    nn.InstanceNorm2d(next_dim),
                    nn.PReLU(next_dim),
                )
            )
            self.model.append(
                BaseResLayer(
                    [
                        nn.Identity(),
                        nn.Sequential(
                            PReLU_Conv2d(next_dim, next_dim, 3, 1, 1),
                            nn.InstanceNorm2d(next_dim),
                        ),
                    ]
                )
            )
            self.model.append(nn.PReLU(next_dim))
            iter_dim = iter_dim // 2
            cur_dim = next_dim
        self.model.append(PReLU_Conv2d(filter_dim, color_channels, 3, 1, 1))
        self.model.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.model)

    def forward(self, z):
        z = z.view(-1, self.z_dim, 1, 1)
        return self.model(z)
