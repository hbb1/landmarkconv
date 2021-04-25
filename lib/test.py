import torch
import numpy as np
import pdb
from torch.autograd import gradcheck
from torch.autograd import Function
import torch.nn as nn
from layers.conv4 import TopLeftPool, TopRightPool, BottomLeftPool, BottomRightPool
from layers.conv8 import I1Pool, I2Pool, I3Pool, I4Pool, I5Pool, I6Pool, I7Pool, I8Pool

var = torch.tensor([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,1,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
]).unsqueeze(0).unsqueeze(0).to(dtype=torch.double)


guide = torch.ones_like(var)
var.requires_grad = True
guide.requires_grad = True
y = I4Pool()(var.cuda(), guide.cuda())


grads={}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook
y.register_hook(save_grad('y'))
var.register_hook(save_grad('x'))
guide.register_hook(save_grad('g'))
y.sum().backward()

input = torch.randn(4, 4, 8,8,dtype=torch.double,requires_grad=True).cuda()
guide = torch.sigmoid(torch.randn(4, 1, 8,8,dtype=torch.double,requires_grad=True)).cuda()
test = gradcheck(lambda x, y: I3Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
print(test)
test = gradcheck(lambda x, y: I4Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
print(test)
test = gradcheck(lambda x, y: I5Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
print(test)
test = gradcheck(lambda x, y: I6Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
print(test)
test = gradcheck(lambda x, y: I7Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
print(test)
test = gradcheck(lambda x, y: I8Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
print(test)
test = gradcheck(lambda x, y: I1Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
print(test)
test = gradcheck(lambda x, y: I2Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
print(test)
# test = gradcheck(lambda x, y: I3Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
# print(test)
test = gradcheck(lambda x, y: I4Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
print(test)
# test = gradcheck(lambda x, y: BottomLeftPool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
# print(test)
# test = gradcheck(lambda x, y: BottomRightPool()(x, y), (input, guide), eps=1e-6, raise_exception=True)