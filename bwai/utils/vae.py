import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder_simple
from ..nn.linear import SLinear
from .generator import G_simple


class VAE(nn.Module):
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

    def loss_func(self, x, x_pred, mean, log_var):
        loss_x = (
            F.mse_loss(x, x_pred, reduction='none')
            .view(x.size(0), -1)
            .mean(dim=1)
        )
        loss_z = (0.5 * (mean**2 + torch.exp(log_var) - log_var - 1)).mean(dim=1)
        loss = loss_x * 10 + loss_z
        loss = loss.mean()
        return loss
    
    def predict(self, batch_size, device):
        z = torch.randn(batch_size, self.latent_dim).to(device)
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        mean, log_var = self.fc_mean(encoded), self.fc_log_var(encoded)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var


class VAE_cluster:
    def __init__(self):
        pass
