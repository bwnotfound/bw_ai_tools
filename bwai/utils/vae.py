import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder_simple
from ..nn.linear import SLinear
from ..nn.conv import SConv2d, SConvTranspose2d
from .generator import G_simple
from .quantization import VectorQuantizer
from .resnet import BaseResLayer


class Scaler(nn.Module):
    """
    Used by VAE. Implementation of https://kexue.fm/archives/7381.
    When VAE take parameter kl_use_bn=True, it will use this class to scale the mean and log_var.
    However, it is discouraged to use this method when KL divergence vanishing is not occurred.
    So the common practice is to use this method in condition when q(x|z) is too strong that a
        little change of z will inject a big noise in x. In this case, KL vanishing is likely to come for z was abondoned,
        which cause the degeneration of VAE to a simple autoencoder.
    Recommonded use case: nlp"""

    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau
        self.theta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, shift=False):
        if not shift:
            scale = torch.sigmoid(self.theta) * self.tau
        else:
            scale = 1 - torch.sigmoid(self.theta) * self.tau
        return x * torch.sqrt(scale)


class VAE(nn.Module):
    r"""
    Implementation of https://kexue.fm/archives/5887 in pytorch
    """

    def __init__(
        self,
        img_size,
        color_channels,
        filter_dim,
        z_dim,
        latent_dim=1,
        min_size=5,
        ffn_dim=512,
        dropout=0.1,
        kl_use_bn=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.color_channels = color_channels
        self.filter_dim = filter_dim
        self.z_dim = z_dim
        self.min_size = min_size
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.kl_use_bn = kl_use_bn

        if kl_use_bn:
            self.kl_bn = nn.BatchNorm1d(latent_dim, affine=False)
            self.scaler = Scaler()

        self.fc_mean = SLinear(z_dim, latent_dim)
        self.fc_log_var = SLinear(z_dim, latent_dim)
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean)
        return epsilon * std + mean

    def get_encoder(self):
        return Encoder_simple(
            self.img_size,
            self.color_channels,
            self.filter_dim,
            self.z_dim,
            self.min_size,
            self.ffn_dim,
            self.dropout,
        )

    def get_decoder(self):
        return G_simple(
            self.img_size,
            self.latent_dim,
            self.filter_dim,
            self.color_channels,
        )

    def loss_func(self, x, packs):
        x_pred, mean, log_var = packs
        loss_x = F.mse_loss(x, x_pred, reduction='sum') / x.size(0)
        loss_z = (
            (0.5 * (mean**2 + torch.exp(log_var) - log_var - 1)).sum(dim=1).mean()
        )
        loss = loss_x + loss_z
        return loss

    def predict(self, batch_size, device):
        z = torch.randn(batch_size, self.latent_dim).to(device)
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        mean, log_var = self.fc_mean(encoded), self.fc_log_var(encoded)
        if self.kl_use_bn:
            mean = self.scaler(self.kl_bn(mean))
            log_var = self.scaler(self.kl_bn(log_var), shift=True)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var


class VAE_cluster(nn.Module):
    r"""
    Implementation of https://kexue.fm/archives/5887 in pytorch
    """

    def __init__(
        self,
        img_size,
        color_channels,
        filter_dim,
        z_dim,
        num_classes,
        latent_dim=1,
        min_size=5,
        ffn_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.color_channels = color_channels
        self.filter_dim = filter_dim
        self.z_dim = z_dim
        self.min_size = min_size
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.fc_mean = SLinear(z_dim, latent_dim)
        self.fc_log_var = SLinear(z_dim, latent_dim)
        self.fc_z2y = nn.Sequential(
            SLinear(latent_dim, num_classes), nn.Softmax(dim=-1)
        )
        self.fc_y2z = SLinear(num_classes, latent_dim)

        self.y2z_mean = nn.Parameter(
            torch.zeros(num_classes, latent_dim), requires_grad=True
        )  # q(z|y): arange然后embedding等效于直接parameter一个tensor
        self.q_y = nn.Parameter(
            torch.ones(num_classes) / num_classes, requires_grad=False
        )
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean)
        return epsilon * std + mean

    def get_encoder(self):
        return Encoder_simple(
            self.img_size,
            self.color_channels,
            self.filter_dim,
            self.z_dim,
            self.min_size,
            self.ffn_dim,
            self.dropout,
        )

    def get_decoder(self):
        return G_simple(
            self.img_size,
            self.latent_dim,
            self.filter_dim,
            self.color_channels,
        )

    def loss_func(self, x, packs):
        x_pred, log_var, z, y = packs  # y: [b, num_classes]
        batch_size = x.size(0)
        loss_x = F.mse_loss(x, x_pred, reduction='sum') / batch_size
        loss_y = (
            F.binary_cross_entropy_with_logits(
                y, self.q_y.unsqueeze(0).expand_as(y), reduction='sum'
            )
            / batch_size
        )  # TODO: 这里暂时先简单粗暴的用binary_cross_entropy_with_logits近似KL散度
        loss_cluster = -0.5 * (
            log_var.sum(dim=1)
            - (
                y * ((z.unsqueeze(1) - self.y2z_mean.unsqueeze(0)) ** 2).sum(dim=-1)
            ).sum(dim=1)
        )  # TODO: mse_loss在广播时会warning同时两个都需要广播，所以直接用**2。测试结果是性能损失不大
        loss_cluster = loss_cluster.mean()
        loss = loss_x + loss_y + loss_cluster
        return loss

    def predict(self, batch_size, device=None, category=None, std=1):
        if device is None:
            device = self.y2z_mean.device
        if category is None:
            category = torch.randint(0, self.num_classes, (1,)).item()
        z = torch.randn(batch_size, self.latent_dim).to(device) * std + self.y2z_mean[
            category
        ].unsqueeze(0)
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        mean, log_var = self.fc_mean(encoded), self.fc_log_var(encoded)
        z = self.reparameterize(mean, log_var)
        y = self.fc_z2y(z)

        return self.decoder(z), log_var, z, y

# TODO: 需要测试num_heads对特征纠缠的解耦程度。预测会有一定的解耦效果，但还未测试
class VQ_VAE(nn.Module):
    # Implementation of https://kexue.fm/archives/6760 . Which is actually a VQ-AE
    def __init__(
        self,
        img_size,
        color_channels,
        z_dim,
        codebook_size,
        num_heads=None,
        z_q_beta=4.0,  # z_q_beta表示训练力度之比: codebook/encoder。建议大于1，因为希望z_q去接近z而不是z去接近z_q
        x_gama=1.0,  # x_gama表示训练力度之比: decoder/other parameter
    ):
        super().__init__()
        self.img_size = img_size
        self.color_channels = color_channels
        self.z_dim = z_dim
        if num_heads == None:
            num_heads = z_dim // 4
        assert z_dim % num_heads == 0
        self.num_heads = num_heads
        self.z_q_beta = z_q_beta
        self.x_gama = x_gama
        self.num_layers = 2

        self.vq = VectorQuantizer(z_dim, num_heads, codebook_size)

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def get_resnet(self, num_layers=2):
        return nn.Sequential(
                *[
                    nn.Sequential(BaseResLayer(
                        [
                            nn.Identity(),
                            nn.Sequential(
                                SConv2d(self.z_dim, self.z_dim, 3, 1, 1),
                                nn.BatchNorm2d(self.z_dim), 
                            )
                        ]
                    ), nn.PReLU(self.z_dim))
                    for _ in range(num_layers)
                ]
            )

    def get_encoder(self):
        return nn.Sequential(
            SConv2d(self.color_channels, self.z_dim, 3, 2, 1),
            nn.BatchNorm2d(self.z_dim),
            nn.PReLU(self.z_dim),
            SConv2d(self.z_dim, self.z_dim, 3, 2, 1),
            nn.BatchNorm2d(self.z_dim),
            nn.PReLU(self.z_dim),
            self.get_resnet(self.num_layers),
            nn.Conv2d(self.z_dim, self.z_dim, 3, 1, 1),
        )

    def get_decoder(self):
        return nn.Sequential(
            self.get_resnet(self.num_layers),
            SConvTranspose2d(self.z_dim, self.z_dim, 4, 2, 1),
            nn.BatchNorm2d(self.z_dim),
            nn.PReLU(self.z_dim),
            SConvTranspose2d(self.z_dim, self.color_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def loss_func(self, x, packs):
        x_pred, z, z_q = packs
        loss_x = F.mse_loss(x, x_pred, reduction='sum')
        loss_z = F.mse_loss(
            z, z_q.detach(), reduction='sum'
        ) + self.z_q_beta * F.mse_loss(z.detach(), z_q, reduction='sum')
        loss = loss_x * self.x_gama + loss_z
        loss = loss / x.size(0)
        return loss

    def forward(self, x):
        # use straight-through estimator, which is a trick to calculate the gradient even it is not backwardable.
        z = self.encode(x)
        z_q, _ = self.vq(z, which_dim=1)
        return self.decode(z + (z_q - z).detach()), z, z_q
