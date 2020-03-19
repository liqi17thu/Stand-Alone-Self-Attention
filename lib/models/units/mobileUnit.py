import torch.nn as nn

from .activation import build_activation, Hswish
from .utils import get_same_padding
from .seUnit import SEModule
from .saUnit import SAConv
from lib.config import cfg


class MBInvertedConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6,
                 mid_channels=None, use_se=False, act_func='relu6', sa=False, logger=None):
        super(MBInvertedConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        self.inverted_bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim),
            build_activation(self.act_func, inplace=True),
        )

        pad = get_same_padding(self.kernel_size)
        if sa:
            depth_conv_modules = [nn.AvgPool2d(kernel_size, padding=pad, stride=stride)]
        else:
            depth_conv_modules = [nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)]
        depth_conv_modules.append(nn.BatchNorm2d(feature_dim))
        depth_conv_modules.append(build_activation(self.act_func, inplace=True))
        if self.use_se:
            depth_conv_modules.append(SEModule(feature_dim))
        self.depth_conv = nn.Sequential(*depth_conv_modules)

        self.point_linear = nn.Sequential(
            nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.inverted_bottleneck(x)
        out = self.depth_conv(out)
        out = self.point_linear(out)
        if self.use_res_connect:
            out += x
        return out
