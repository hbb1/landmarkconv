import torch
import numpy as np
import pdb
from torch.autograd import gradcheck
from torch.autograd import Function
import torch.nn as nn
from landmarkconv import _C


__all__ = ["PConv2d8", "I1Pool", "I2Pool", "I3Pool", "I4Pool", "I5Pool", "I6Pool", "I7Pool", ]

class I1PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.I1_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.I1_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide


class I2PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.I2_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.I2_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide


class I3PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.I3_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.I3_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide


class I4PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.I4_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.I4_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class I5PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.I5_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.I5_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide



class I6PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.I6_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.I6_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide



class I7PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.I7_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.I7_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide


class I8PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.I8_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.I8_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide


class I1Pool(nn.Module):
    def forward(self, x, guide=None):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        # guide = guide.expand_as(x).contiguous()
        return I1PoolFunction.apply(x, guide)


class I2Pool(nn.Module):
    def forward(self, x, guide=None):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        # guide = guide.expand_as(x).contiguous()
        return I2PoolFunction.apply(x, guide)

class I3Pool(nn.Module):
    def forward(self, x, guide=None):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        # guide = guide.expand_as(x).contiguous()
        return I3PoolFunction.apply(x, guide)

class I4Pool(nn.Module):
    def forward(self, x, guide=None):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        # guide = guide.expand_as(x).contiguous()
        return I4PoolFunction.apply(x, guide)

class I5Pool(nn.Module):
    def forward(self, x, guide=None):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        # guide = guide.expand_as(x).contiguous()
        return I5PoolFunction.apply(x, guide)

class I6Pool(nn.Module):
    def forward(self, x, guide=None):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        # guide = guide.expand_as(x).contiguous()
        return I6PoolFunction.apply(x, guide)

class I7Pool(nn.Module):
    def forward(self, x, guide=None):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        # guide = guide.expand_as(x).contiguous()
        return I7PoolFunction.apply(x, guide)


class I8Pool(nn.Module):
    def forward(self, x, guide=None):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        # guide = guide.expand_as(x).contiguous()
        return I8PoolFunction.apply(x, guide)


class PConv2d8(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # according to the original papr, there are no gates.
        # self.gates = nn.Parameter(torch.ones(8, args[0]).view(8,-1, 1, 1))
        self.pools = [I1Pool(), I2Pool(), I3Pool(), I4Pool(), I5Pool(), I6Pool(), I7Pool(), I8Pool()]

    def forward(self, x):
        x = torch.cat([self.pools[i](x.chunk(8, 1)[i]) for i in range(8)], dim=1)
        x = super().forward(x)