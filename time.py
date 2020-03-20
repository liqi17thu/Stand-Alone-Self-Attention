import torch
import torch.nn as nn
from lib.models.units.saUnit import SAConv
from torch.autograd import Variable

import time



# input
x = torch.rand(3, 160, 32, 32)

# 4 operation
saconv = SAConv(160, 160, kernel_size=3, heads=8, padding=1)
conv = nn.Conv2d(160, 160, 3, padding=1)
spconv = nn.Conv2d(160, 160, 3, padding=1, groups=160)
pooling = nn.AvgPool2d(3, padding=1)

xx = Variable(x.data.clone(), requires_grad=True)
s = time.time()
for _ in range(10):
    xx = saconv(xx)
e = time.time()
print(f'attention forward: {e-s}s')

s = time.time()
xx = xx.mean(3).mean(2).mean(1).mean(0)
xx.backward()
e = time.time()
print(f'attention backward: {e-s}s')

xx = Variable(x.data.clone(), requires_grad=True)
s = time.time()
for _ in range(10):
    xx = conv(xx)
e = time.time()
print(f'conv forward: {e-s}s')

s = time.time()
xx = xx.mean(3).mean(2).mean(1).mean(0)
xx.backward()
e = time.time()
print(f'conv backward: {e-s}s')

xx = Variable(x.data.clone(), requires_grad=True)
s = time.time()
for _ in range(100):
    xx = spconv(xx)
e = time.time()
print(f'spconv forward: {e-s}s')

s = time.time()
xx = xx.mean(3).mean(2).mean(1).mean(0)
xx.backward()
e = time.time()
print(f'spconv backward: {e-s}s')


xx = Variable(x.data.clone(), requires_grad=True)
s = time.time()
for _ in range(100):
    xx = pooling(xx)
e = time.time()
print(f'pooling forward: {e-s}s')

s = time.time()
xx = xx.mean(3).mean(2).mean(1).mean(0)
xx.backward()
e = time.time()
print(f'pooling backward: {e-s}s')