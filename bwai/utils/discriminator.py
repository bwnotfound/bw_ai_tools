import torch.nn as nn
from .resnet import BaseResLayer as ResLayer


class D_Common(nn.Module):
    def __init__(self, img_size, color_channels, filter_dim):
        super().__init__()
        self.model = nn.ModuleList()
        cur_dim = filter_dim
        self.model.append(
            nn.Sequential(
                nn.Conv2d(color_channels, cur_dim, 5, 1, 2),
                nn.GroupNorm(4, cur_dim),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 1),
            )
        )
        cur_size = img_size // 2 + img_size % 2
        for _ in range(2):
            self.model.append(
                ResLayer(
                    (
                        nn.Identity(),
                        nn.Sequential(
                            nn.Conv2d(cur_dim, cur_dim // 4, 1),
                            nn.GroupNorm(4, cur_dim // 4),
                            nn.ReLU(),
                            nn.Conv2d(cur_dim // 4, cur_dim // 4, 3, 1, 1),
                            nn.GroupNorm(4, cur_dim // 4),
                            nn.ReLU(),
                            nn.Conv2d(cur_dim // 4, cur_dim, 1),
                            nn.GroupNorm(4, cur_dim),
                        ),
                    )
                )
            )
            self.model.append(nn.ReLU())
        while cur_size > 3:
            next_dim = cur_dim * 2
            self.model.append(nn.AvgPool2d(2, 2))
            self.model.append(
                ResLayer(
                    (
                        nn.Sequential(
                            nn.Conv2d(cur_dim, next_dim, 1),
                            nn.GroupNorm(4, next_dim),
                        ),
                        nn.Sequential(
                            nn.Conv2d(cur_dim, next_dim // 4, 1),
                            nn.GroupNorm(4, next_dim // 4),
                            nn.ReLU(),
                            nn.Conv2d(next_dim // 4, next_dim // 4, 3, 1, 1),
                            nn.GroupNorm(4, next_dim // 4),
                            nn.ReLU(),
                            nn.Conv2d(next_dim // 4, next_dim, 1),
                            nn.GroupNorm(4, next_dim),
                        ),
                    )
                )
            )
            self.model.append(nn.ReLU())
            cur_dim = next_dim
            cur_size = cur_size // 2 + cur_size % 2

        self.model.append(
            nn.Sequential(
                nn.AvgPool2d(cur_size),
                nn.Flatten(),
                nn.Linear(cur_dim, cur_dim),
                nn.ReLU(),
                nn.Linear(cur_dim, cur_dim),
                nn.ReLU(),
                nn.Linear(cur_dim, 1),
            )
        )
        self.model = nn.Sequential(*self.model)
        
        
        self.model = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(color_channels, filter_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(filter_dim) x 32 x 32``
            nn.Conv2d(filter_dim, filter_dim * 2, 4, 2, 1),
            nn.InstanceNorm2d(filter_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(filter_dim*2) x 16 x 16``
            nn.Conv2d(filter_dim * 2, filter_dim * 4, 4, 2, 1),
            nn.InstanceNorm2d(filter_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(filter_dim*4) x 8 x 8``
            nn.Conv2d(filter_dim * 4, filter_dim * 8, 4, 2, 1),
            nn.InstanceNorm2d(filter_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(filter_dim*8) x 4 x 4``
            nn.Conv2d(filter_dim * 8, 1, 4, 1, 0),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x)


