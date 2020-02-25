import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

from .utils import get_same_padding
from .activation import Hswish

class SAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, heads=1, bias=False):
        super(SAConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.heads = heads

        assert self.out_channels % self.heads == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, groups=heads)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, groups=heads)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, groups=heads)
        self.activation = Hswish()

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        v_out = self.activation(v_out)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.heads, self.out_channels // self.heads, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.heads, self.out_channels // self.heads, height, width, -1)

        q_out = q_out.view(batch, self.heads, self.out_channels // self.heads, height, width, 1)

        out = (q_out * k_out).sum(dim=2, keepdim=True) * np.sqrt((self.heads // self.out_channels))
        out = F.softmax(out, dim=-1)
        out = (out * v_out).sum(dim=-1)
        out = out.view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class SAPooling(nn.Module):
    def __init__(self, channels, heads=1, bias=False):
        super(SAConv, self).__init__()
        self.channels = channels
        self.heads = heads

        assert self.out_channels % self.heads == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.query = nn.Parameter(torch.randn(1, heads, channels // heads), requires_grad=True)
        self.key_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=bias, groups=heads)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=bias, groups=heads)
        self.activation = Hswish()

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        k_out = self.key_conv(x)
        v_out = self.value_conv(x)
        v_out = self.activation(v_out)

        k_out = k_out.view(batch, self.heads, self.out_channels // self.heads, height, width)
        k_out = k_out.permute(2, 0, 1, 3, 4).contiguous()
        k_out = k_out.view(self.out_channels // self.heads, -1)

        v_out = v_out.view(batch, self.heads, self.out_channels // self.heads, height, width)
        v_out = v_out.permute(0, 1, 3, 4, 2).contiguous()
        v_out = v_out.view(-1, self.out_channels // self.heads)

        q_out = self.query.repeat(batch, 1, 1)
        q_out = q_out.view(-1, self.channels // self.heads)

        out = torch.mm(q_out, k_out) * np.sqrt((self.heads // self.out_channels))
        out = F.softmax(out, dim=-1)
        out = torch.mm(out, v_out)
        out = out.view(batch, self.channels, 1, 1)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.query, 0, 1)


class SAStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(SAStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)


class SABottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7, groups=1, base_width=64, heads=8):
        super(SABottleneck, self).__init__()
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
            SAConv(width, width, kernel_size=self.kernel_size, padding=padding, heads=self.heads),
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
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))

        out += self.shortcut(x)
        out = F.relu(out)

        return out

