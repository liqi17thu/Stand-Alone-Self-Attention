import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import get_same_padding


class DynamicConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, heads=1, bias=False, with_conv=False):
        super(DynamicConv, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.heads = heads

        assert self.channels % self.heads == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.filter = nn.Parameter(torch.randn(channels, kernel_size * kernel_size * heads), requires_grad=True)

        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        stride_x = x[:, :, ::self.stride, ::self.stride]
        sep_filter = stride_x.permute(0, 2, 3, 1).contiguous().view(batch * height // self.stride * width // self.stride, channels).mm(self.filter)
        sep_filter = sep_filter.view(batch, height // self.stride, width // self.stride, self.kernel_size, self.kernel_size, self.heads)
        sep_filter = sep_filter.unsqueeze(-1).repeat(1, 1, 1, 1, 1, 1, channels // self.heads)
        sep_filter = sep_filter.view(batch, height // self.stride, width // self.stride, self.kernel_size, self.kernel_size, channels)
        # filter shape: B, H/s, W/s, K, K, C

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        padded_x = padded_x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        padded_x = padded_x.permute(0, 2, 3, 4, 5, 1).contiguous()
        # padded_x shape: B, H/s, W/s, K, K, C

        out = (padded_x * sep_filter).sum(4).sum(3)
        out = out.permute(0, 3, 1, 2).contiguous()

        if self.with_conv:
            out += self.conv(x)

        return out

    def reset_parameters(self):
        if self.with_conv:
            init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.filter, 0, 1)


class DynamicBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7, groups=1, base_width=64, heads=8, with_conv=False):
        super(DynamicBottleneck, self).__init__()
        self.stride = stride
        self.heads = heads
        self.kernel_size = kernel_size

        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )

        padding = get_same_padding(kernel_size)
        self.conv2 = nn.Sequential(
            DynamicConv(width, kernel_size=self.kernel_size, stride=self.stride, padding=padding, heads=self.heads, with_conv=with_conv),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out
