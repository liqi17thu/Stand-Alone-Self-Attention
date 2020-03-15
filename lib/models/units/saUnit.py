import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg

class SABottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, groups=1, expansion=4,
                 base_width=64, heads=8):
        super(SABottleneck, self).__init__()
        self.stride = stride
        self.heads = heads
        self.kernel_size = kernel_size
        self.with_conv = cfg.model.with_conv
        self.expansion = expansion

        width = int(out_channels * (base_width / 64.)) * groups


        self.conv1 = nn.Sequential(
            nn.AvgPool2d(7, padding=3),
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=width),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool2d(7, padding=3),
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x, r=None):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out
