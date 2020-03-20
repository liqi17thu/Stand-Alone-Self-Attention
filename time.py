import torch
import torch.nn as nn
from lib.models.units.saUnit import SAConv
from torch.autograd import Variable
from lib.models.units.utils import get_same_padding
from lib.config import cfg
from lib.utils import get_logger, get_net_info

import os

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


def time_counting(x, Net, kernel=3, heads=8, gpu=False, dryrun=False, logger=None):
    xx = Variable(x.data.clone(), requires_grad=True)
    b, c, h, w = xx.shape
    pad = get_same_padding(kernel)
    net = Net(c, c, kernel, pad, heads)

    if gpu:
        xx = xx.cuda()
        net = net.cuda()

    if not dryrun:
        logger.info(f'{net.__class__.__name__}:')
    if not gpu:
        logger.info('forward:')
        with torch.autograd.profiler.profile() as prof:
            for _ in range(10):
                xx = net(xx)
        logger.info(prof.key_averages().table(sort_by="self_cpu_time_total"))

        logger.info('backward:')
        with torch.autograd.profiler.profile() as prof:
            xx = xx.mean(3).mean(2).mean(1).mean(0)
            xx.backward()
        logger.info(prof.key_averages().table(sort_by="self_cpu_time_total"))
    else:
        if not dryrun:
            logger.info('forward:')
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            for _ in range(100):
                xx = net(xx)
        if not dryrun:
            logger.info(prof.key_averages().table(sort_by="cuda_time"))

        if not dryrun:
            logger.info('backward:')
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            xx = xx.mean(3).mean(2).mean(1).mean(0)
            xx.backward()
        if not dryrun:
            logger.info(prof.key_averages().table(sort_by="cuda_time"))

    get_net_info(net, (160, 32, 32), logger=logger)


logger = get_logger(os.path.join(cfg.save_path, 'net_info.log'), False)

# input
x = torch.rand(3, 160, 32, 32)

if cfg.cuda:
    time_counting(x, SANet, gpu=True, dryrun=True, logger=logger)
    time_counting(x, SANet, gpu=True, logger=logger)
    time_counting(x, ConvNet, gpu=True, logger=logger)
    time_counting(x, SPConvNet, gpu=True, logger=logger)
    time_counting(x, PoolingNet, gpu=True, logger=logger)
else:
    time_counting(x, SANet, logger=logger)
    time_counting(x, ConvNet, logger=logger)
    time_counting(x, SPConvNet, logger=logger)
    time_counting(x, PoolingNet, logger=logger)

