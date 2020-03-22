import torch
import torch.nn as nn
from .utils import get_same_padding
from .activation import build_activation
from .seUnit import SEModule

from lib.config import cfg

class MBInvertedConvLayer(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, se, stride, pooling):
        super(MBInvertedConvLayer, self).__init__()
        self.stride = stride
        self.se = SEModule(out_size) if se else None
        self.nolinear = build_activation(nolinear)
        self.rezero = cfg.model.rezero
        if self.rezero:
            self.scale = nn.Parameter(torch.Tensor([0]))

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)

        pad = get_same_padding(kernel_size)
        if pooling:
            self.conv2 = nn.AvgPool2d(kernel_size, stride, pad)
        else:
            self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                                   padding=pad, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear(self.bn1(self.conv1(x)))
        out = self.nolinear(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        if self.rezero:
            out = out * self.scale + self.shortcut(x)
        else:
            out += self.shortcut(x)
        return out
