import torch
import torch.nn as nn
import math

"""
    He initialization, also called kaiming initialization. Need to be used with prelu. 
    Otherwise use relu with a=0. 
    It worth notice that you should warm up some epochs in larger learning rate when you use this initialization.
    Because the value of the output is too small in the beginning which leads the gradient to be small in response.
"""
def he_normal_(tensor: torch.Tensor, a=0.25):
    std = math.sqrt(2.0 / ((1 + a**2) * (tensor[0][0].numel() * tensor.shape[1])))
    nn.init.normal_(tensor, 0, std)


def he_uniform_(tensor: torch.Tensor, a=0.25):
    var = 2.0 / ((1 + a**2) * (tensor[0][0].numel() * tensor.shape[1]))
    l = math.sqrt(3 * var)
    nn.init.uniform_(tensor, -l, l)


# TODO: Still wait for test on grad calculation in fp16.
"""
    He initialization, also called kaiming initialization. Need to be used with prelu. 
    Otherwise use relu with a=0. 
    It worth notice that you should warm up some epochs in larger learning rate when you use this initialization.
    Because the value of the output is too small in the beginning which leads the gradient to be small in response.
"""
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''

    def __init__(self, name, a):
        self.name = name
        self.a = a

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / (fan_in * (1 + self.a**2)))

    @staticmethod
    def apply(module, name, a=0.25):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name, a)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# Quick apply for scaled weight
def quick_scale(module, name='weight', a=0.25):
    ScaleW.apply(module, name, a)
    return module
