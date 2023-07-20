import torch
import torch.nn as nn


class BaseResLayer(nn.Module):
    def __init__(self, models):
        super().__init__()
        if isinstance(models, (list, tuple)):
            models = nn.ModuleList(models)
        elif not isinstance(models, nn.ModuleList):
            raise ValueError("models must be a list or tuple or nn.ModuleList")
        self.models = models

    def forward(self, x):
        res = 0
        for model in self.models:
            res = res + model(x)
        return res