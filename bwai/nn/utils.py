import torch
import torch.nn as nn

# class DtypeShift(nn.Module):
#     def __init__(self, dtype) -> None:
#         super().__init__()
#         self.dtype = dtype
        
# class DtypeShiftWrap(nn.Module):
#     def __init__(self, module, pre_dtype=None, post_dtype=None):
#         super().__init__()
#         self.module = module
#         self.pre_dtype = pre_dtype
#         self.post_dtype = post_dtype
        
#     def forward(self, x):
#         if self.pre_dtype is not None:
#             x = x.to(self.pre_dtype)
#         output = self.module(x)
#         if self.post_dtype is not None:
#             output = output.to(self.post_dtype)
#         return output
    
class PixelShuffle(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x.reshape(
            x.shape[0],
            x.shape[1] // (self.scale_factor**2),
            x.shape[2] * self.scale_factor,
            x.shape[3] * self.scale_factor,
        )