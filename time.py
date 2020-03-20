import torch
import torch.nn as nn
from lib.models.units.saUnit import SAConv
from torch.autograd import Variable
from lib.models.units.utils import get_same_padding

import time

class SANet(nn.Module):
    def __init__(self, inplanes, planes, kernel, padding, heads):
        super(SANet, self).__init__()
        self.saconv = SAConv(inplanes, planes, kernel_size=kernel, padding=padding, heads=heads)

    def forward(self, x):
        return self.saconv(x)


class ConvNet(nn.Module):
    def __init__(self, inplanes, planes, kernel, padding, heads):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel, padding=padding)

    def forward(self, x):
        return self.conv(x)


class SPConvNet(nn.Module):
    def __init__(self, inplanes, planes, kernel, padding, heads):
        super(SPConvNet, self).__init__()
        self.spconv = nn.Conv2d(inplanes, planes, kernel_size=kernel, padding=padding, groups=planes)

    def forward(self, x):
        return self.spconv(x)


class PoolingNet(nn.Module):
    def __init__(self, inplanes, planes, kernel, padding, heads):
        super(PoolingNet, self).__init__()
        self.pooling = nn.AvgPool2d(kernel, padding=padding)

    def forward(self, x):
        return self.pooling(x)


def time_counting(x, Net, kernel=3, heads=8, gpu=False):
    xx = Variable(x.data.clone(), requires_grad=True)
    b, c, h, w = xx.shape
    pad = get_same_padding(kernel)
    net = Net(c, c, kernel, pad, heads)

    if gpu:
        xx = xx.cuda()
        net = net.cuda()

    # s = time.time()
    # for _ in range(100):
    #     xx = net(xx)
    # e = time.time()
    # print(f'{net.__class__.__name__} forward: {(e - s):.2E}s')

    # s = time.time()
    # xx = xx.mean(3).mean(2).mean(1).mean(0)
    # xx.backward()
    # e = time.time()
    # print(f'{net.__class__.__name__} backward: {(e - s):.2E}s')

    print(f'{net.__class__.__name__}:')
    print('forward:')
    with torch.autograd.profiler.profile() as prof:
        for _ in range(10):
            xx = net(xx)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    print('backward:')
    with torch.autograd.profiler.profile() as prof:
        xx = xx.mean(3).mean(2).mean(1).mean(0)
        xx.backward()
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


# input
x = torch.rand(3, 160, 32, 32)

time_counting(x, SANet)