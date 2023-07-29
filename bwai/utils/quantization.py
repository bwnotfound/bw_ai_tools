import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantization(nn.Module):
    def __init__(self, dim, codebook_size):
        super(VectorQuantization, self).__init__()
        self.dim = dim
        self.codebook_size = codebook_size

        self.embedding = nn.Parameter(
            torch.randn(codebook_size, dim), requires_grad=True
        )

    def preprocess(self, x):
        assert len(x.shape) > 0 and x.shape[-1] == self.dim
        if len(x.shape) == 1:
            x.unsqueeze_(0)
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])
        return x

    def postprocess_emb(self, x, shape):
        return x.view(*shape[:-1])

    def quantize(self, x):
        e_t = self.embedding.t()
        dist = -(
            x.pow(2).sum(dim=-1, keepdim=True)
            - 2 * x @ e_t
            + e_t.pow(2).sum(dim=0, keepdim=True)
        )  # 该运算会比直接用广播算(x-y)^2快，约为原方法耗时的40%
        embed_indicies = torch.argmax(dist, dim=1)
        return embed_indicies

    def dequantize(self, embed_indicies):
        return F.embedding(embed_indicies, self.embedding)

    def encode(self, x, which_dim=-1):
        if which_dim != -1:
            x = x.permute(
                *range(which_dim), *range(which_dim + 1, len(x.shape)), which_dim
            )
        shape = x.shape
        x = self.preprocess(x)
        embed_indicies = self.quantize(x)
        embed_indicies = self.postprocess_emb(embed_indicies, shape)
        return embed_indicies

    def decode(self, embed_indices, which_dim=-1):
        quantization = self.dequantize(embed_indices)
        if which_dim != -1:
            quantization = quantization.permute(
                *range(which_dim), -1, *range(which_dim, len(quantization.shape) - 1)
            )
        return quantization

    def forward(self, x, which_dim=-1):
        embed_indicies = self.encode(x, which_dim=which_dim)
        quantization = self.decode(embed_indicies, which_dim=which_dim)
        return quantization, embed_indicies


class VectorQuantizer(nn.Module):
    
    r"""
        Vector Quantization(VQ) Module with feature of Product quantization(PQ).
        It's test that PQ can improve the performance of VQ enormously.
        It's test that use asynic has negative effect on performance.
    """
    
    def __init__(self, dim, num_heads, codebook_size):
        super(VectorQuantizer, self).__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.codebook_size = codebook_size

        self.vq_list = nn.ModuleList(
            [
                VectorQuantization(dim // num_heads, codebook_size)
                for _ in range(num_heads)
            ]
        )

    def encode(self, x, which_dim=-1):
        if which_dim != -1:
            x = x.permute(
                *range(which_dim), *range(which_dim + 1, len(x.shape)), which_dim
            )
        x_stacks = x.chunk(self.num_heads, dim=-1)
        embed_indicies = torch.stack(
            [
                self.vq_list[i].encode(x_stacks[i])
                for i in range(self.num_heads)
            ],
            dim=-1,
        )
        return embed_indicies

    def decode(self, embed_indices, which_dim=-1):
        embed_indices_stacks = [e.squeeze(-1) for e in embed_indices.chunk(self.num_heads, dim=-1)]
        quantization = torch.cat(
            [
                self.vq_list[i].decode(embed_indices_stacks[i])
                for i in range(self.num_heads)
            ],
            dim=-1,
        )
        if which_dim != -1:
            quantization = quantization.permute(
                *range(which_dim), -1, *range(which_dim, len(quantization.shape) - 1)
            )
        return quantization

    def forward(self, x, which_dim=-1):
        embed_indices = self.encode(x, which_dim=which_dim)
        quantization = self.decode(embed_indices, which_dim=which_dim)
        return quantization, embed_indices