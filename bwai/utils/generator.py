import torch
import torch.nn as nn
from .resnet import BaseResLayer


# class BaseProgressiveGenerator(nn.Module):
#     def __init__(self, z_dim, filter_dim, num_layers, color_channel):
#         super().__init__()
#         self.num_layers = num_layers
#         self.color_channel = color_channel
#         self.filter_dim = filter_dim
#         self.z_dim = z_dim

#         self.model = nn.ModuleList()


class BaseGenerator(nn.Module):
    r"""
    img_size: 2**(depthModel + 1)
    """

    def __init__(self, z_dim, filter_dim, num_layers, color_channel):
        super().__init__()
        self.num_layers = num_layers
        self.color_channel = color_channel
        self.filter_dim = filter_dim
        self.z_dim = z_dim

        self.model = nn.ModuleList()
        cur_dim = filter_dim * 2**num_layers
        self.model.append(
            nn.Sequential(
                nn.Conv2d(z_dim, z_dim, 1),
                nn.ReLU(),
            )
        )
        self.model.append(nn.ConvTranspose2d(z_dim, cur_dim, 2, 1, 0))

        for _ in range(num_layers):
            next_dim = cur_dim // 2
            self.model.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            self.model.append(
                BaseResLayer(
                    [
                        nn.Sequential(
                            nn.Conv2d(cur_dim, next_dim, 1),
                            nn.GroupNorm(4, next_dim),
                        ),
                        nn.Sequential(
                            nn.Conv2d(cur_dim, next_dim, 3, 1, 1),
                            nn.GroupNorm(4, next_dim),
                            nn.ReLU(),
                            nn.Conv2d(next_dim, next_dim, 3, 1, 1),
                            nn.GroupNorm(4, next_dim),
                        ),
                    ]
                )
            )
            self.model.append(nn.ReLU())
            cur_dim = next_dim
        for _ in range(1):
            self.model.append(
                nn.Sequential(
                    BaseResLayer(
                        [
                            nn.Identity(),
                            nn.Sequential(
                                nn.Conv2d(cur_dim, cur_dim, 3, 1, 1),
                                nn.GroupNorm(4, cur_dim),
                            ),
                        ]
                    )
                )
            )
            self.model.append(nn.ReLU())
        self.model.append(nn.Conv2d(cur_dim, color_channel, 1))
        self.model.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.model)
        
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( z_dim, filter_dim * 8, 4, 1, 0),
            nn.InstanceNorm2d(filter_dim * 8),
            nn.ReLU(True),
            # state size. ``(filter_dim*8) x 4 x 4``
            nn.ConvTranspose2d(filter_dim * 8, filter_dim * 4, 4, 2, 1),
            nn.InstanceNorm2d(filter_dim * 4),
            nn.ReLU(True),
            # state size. ``(filter_dim*4) x 8 x 8``
            nn.ConvTranspose2d( filter_dim * 4, filter_dim * 2, 4, 2, 1),
            nn.InstanceNorm2d(filter_dim * 2),
            nn.ReLU(True),
            # state size. ``(filter_dim*2) x 16 x 16``
            nn.ConvTranspose2d( filter_dim * 2, filter_dim, 4, 2, 1),
            nn.InstanceNorm2d(filter_dim),
            nn.ReLU(True),
            # state size. ``(filter_dim) x 32 x 32``
            nn.ConvTranspose2d( filter_dim, color_channel, 4, 2, 1),
            nn.Sigmoid()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, z):
        z = z.view(-1, self.z_dim, 1, 1)
        return self.model(z)
