import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .units.activation import Hswish
from .units.mobileUnit import MBInvertedConvLayer
from .units.seUnit import SEModule

from lib.config import cfg


class MobileNetV3(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = Hswish()

        setting = [
           # k,  in,  mid, out,  nl,       se,    s, pool
            [3,  16,  16,  16,  'relu',    False, 1, True],
            [3,  16,  64,  24,  'relu',    False, 2, True],
            [3,  24,  72,  24,  'relu',    False, 1, True],
            [5,  24,  72,  40,  'relu',    True,  2, True],
            [5,  40,  120, 40,  'relu',    True,  1, True],
            [5,  40,  120, 40,  'relu',    True,  1, True],
            [3,  40,  240, 80,  'h_swish', False, 2, True],
            [3,  80,  200, 80,  'h_swish', False, 1, True],
            [3,  80,  184, 80,  'h_swish', False, 1, True],
            [3,  80,  184, 80,  'h_swish', False, 1, True],
            [3,  80,  480, 112, 'h_swish', True,  1, True],
            [3,  112, 672, 112, 'h_swish', True,  1, True],
            [5,  112, 672, 160, 'h_swish', True,  2, True],
            [5,  160, 960, 160, 'h_swish', True,  1, True],
            [5,  160, 960, 160, 'h_swish', True,  1, True],
            ]

        for i in range(cfg.model.num_resblock):
            setting[i][7] = False

        self.bneck = []
        for k, inplane, mid, plane, nl, se, s, pool in setting:
            self.bneck.append(MBInvertedConvLayer(k, inplane, mid, plane, nl, se, s, pool))
        self.bneck = nn.Sequential(*self.bneck)

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = Hswish()
        self.linear3 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(1280)
        self.hs3 = Hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.hs3(self.bn3(self.linear3(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear4(out)
        return out
