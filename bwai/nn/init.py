import torch
import torch.nn as nn
import math

"""
    He initialization, also called kaiming initialization. Need to be used with prelu. 
    Otherwise use relu with a=0. 
    It worth notice that you should warm up some epochs in larger learning rate when you use this initialization.
    Because the value of the output is too small in the beginning which leads the gradient to be small in response.
"""
def he_normal_(tensor: torch.Tensor, a=0.25, dim=0):
    std = math.sqrt(2.0 / ((1 + a**2) * (tensor.numel() // tensor.shape[dim])))
    nn.init.normal_(tensor, 0, std)

# dim: output_channel
def he_uniform_(tensor: torch.Tensor, a=0.25, dim=0):
    var = 2.0 / ((1 + a**2) * (tensor.numel() // tensor.shape[dim]))
    l = math.sqrt(3 * var)
    nn.init.uniform_(tensor, -l, l)


# TODO: Still wait for test on grad calculation in fp16.
"""
    He initialization, also called kaiming initialization. Need to be used with prelu. 
    Otherwise use relu with a=0. 
    It worth notice that you should warm up some epochs in larger learning rate when you use this initialization.
    Because the value of the output is too small in the beginning which leads the gradient to be small in response.
"""
# class ScaleW:
#     '''
#     Constructor: name - name of attribute to be scaled
#     '''

#     def __init__(self, name, a=0.25, regularization=None):
#         self.name = name
#         self.a = a
#         self.regularization = regularization

#     def scale(self, module):
#         weight = getattr(module, self.name + '_orig')
#         fan_in = weight.data.size(1) * weight.data[0][0].numel()

#         return weight * math.sqrt(2 / (fan_in * (1 + self.a**2)))

#     @staticmethod
#     def apply(module, name, a=0.25, regularization=None):
#         '''
#         Apply runtime scaling to specific module
#         '''
#         hook = ScaleW(name, a, regularization)
#         weight = getattr(module, name)
#         module.register_parameter(name + '_orig', nn.Parameter(weight.data))
#         del module._parameters[name]
#         module.register_forward_pre_hook(hook)

#     def __call__(self, module, whatever):
#         weight = self.scale(module)
#         if self.regularization is not None:
#             weight = self.regularization(weight)
#         setattr(module, self.name, weight)


# # Quick apply for scaled weight
# def quick_scale(module, name='weight', a=0.25, regularization=None):
#     ScaleW.apply(module, name, a, regularization)
#     return module
