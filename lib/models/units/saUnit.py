import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import get_same_padding
from .postionalEncoding import PositionalEncoding, SinePositionalEncoding
from .shake_shake import get_alpha_beta, shake_function
from .mixUnit import MixedConv2d
from .shuffleUnit import ChannelShuffle
from lib.config import cfg


class SAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, heads=1, bias=False, r_dim=256,
                 logger=None):
        super(SAConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.heads = heads
        self.encoding = cfg.model.encoding
        self.temperature = cfg.model.temperature
        self.logger = logger

        assert self.out_channels % self.heads == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        if self.encoding != 'none':
            self.encoder = PositionalEncoding(out_channels, kernel_size, heads, bias, self.encoding, r_dim)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, groups=heads)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, stride=stride, groups=heads)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, groups=heads)

        self.reset_parameters()

    def forward(self, x, r=None):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # positonal encoding
        if self.encoding == 'xl':
            k_out, r_out, u, v = self.encoder(k_out, r)
        elif self.encoding == 'none':
            pass
        else:
            k_out = self.encoder(k_out)

        k_out = k_out.contiguous().view(batch, self.heads, self.out_channels // self.heads, height // self.stride,
                                        width // self.stride, -1)
        v_out = v_out.contiguous().view(batch, self.heads, self.out_channels // self.heads, height // self.stride,
                                        width // self.stride, -1)
        q_out = q_out.view(batch, self.heads, self.out_channels // self.heads, height // self.stride,
                           width // self.stride, 1)

        if self.encoding == 'xl':
            out = q_out * k_out + q_out * r_out + u * k_out + v * r_out
        else:
            out = q_out * k_out
        out = out.sum(dim=2, keepdim=True) * self.temperature
        out = F.softmax(out, dim=-1)

        # print attention info
        if cfg.test and cfg.disp_attention and cfg.ddp.local_rank == 0:
            for head in range(self.heads):
                self.logger.info("head {}".format(head))
                for h in range(height // self.stride):
                    for w in range(width // self.stride):
                        self.logger.info("height {} width {}".format(h, w))
                        for k in range(self.kernel_size):
                            loggerInfo = "{:.3f} " * self.kernel_size
                            self.logger.info(loggerInfo.format(
                                *out[0][head][0][h][w][k * self.kernel_size:(k + 1) * self.kernel_size].tolist()))

        out = (out * v_out).sum(dim=-1)
        out = out.view(batch, -1, height // self.stride, width // self.stride)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')


class SAFull(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, heads=1, bias=False, logger=None):
        super(SAFull, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.heads = heads
        self.logger = logger

        assert self.out_channels % self.heads == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.encoder = SinePositionalEncoding(out_channels)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, groups=heads)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias, groups=heads)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, groups=heads)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        q_out = self.query_conv(x)
        k_out = self.key_conv(x)
        v_out = self.value_conv(x)

        # positional embedding
        k_out = k_out.view(batch, self.out_channels, -1)
        k_out = self.encoder(k_out)
        k_out = k_out.view(batch, self.out_channels, height, width)

        q_out = q_out.view(batch, self.heads, self.out_channels // self.heads, height // self.stride,
                           width // self.stride)
        q_out = q_out.permute(0, 1, 3, 4, 2).contiguous()
        q_out = q_out.view(batch * self.heads, -1, self.out_channels // self.heads)

        k_out = k_out.view(batch, self.heads, self.out_channels // self.heads, height, width)
        k_out = k_out.view(batch * self.heads, self.out_channels // self.heads, -1)

        v_out = v_out.view(batch, self.heads, self.out_channels // self.heads, height, width)
        v_out = v_out.permute(0, 1, 3, 4, 2).contiguous()
        v_out = v_out.view(batch * self.heads, -1, self.out_channels // self.heads)

        out = torch.bmm(q_out, k_out)
        out = F.softmax(out, dim=-1)

        # print attention info

        if not self.training and x.get_device() == 1 and cfg.disp_attention:
            temp = out.view(batch, self.heads, height // self.stride, width // self.stride, height, width)
            for head in range(self.heads):
                self.logger.info("head {}".format(head))
                for h in range(height // self.stride):
                    for w in range(width // self.stride):
                        self.logger.info("height {} width {}".format(h, w))
                        for k in range(height):
                            loggerInfo = "{:.3f} " * width
                            self.logger.info(loggerInfo.format(*temp[0][head][h][w][k].tolist()))

        out = torch.bmm(out, v_out)
        out = out.view(batch, self.heads, height // self.stride, width // self.stride, self.out_channels // self.heads)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.view(batch, -1, height // self.stride, width // self.stride)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')


class SAPooling(nn.Module):
    def __init__(self, channels, heads=1, bias=False, logger=None):
        super(SAPooling, self).__init__()
        self.channels = channels
        self.heads = heads
        self.logger = logger
        self.temperature = cfg.model.temperature

        assert self.channels % self.heads == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.query = nn.Parameter(torch.randn(1, heads, channels // heads), requires_grad=True)
        # self.key_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=bias, groups=heads)
        # self.value_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=bias, groups=heads)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        # k_out = self.key_conv(x)
        # v_out = self.value_conv(x)
        k_out = x
        v_out = x

        k_out = k_out.view(batch, self.heads, self.channels // self.heads, -1)
        v_out = v_out.view(batch, self.heads, self.channels // self.heads, -1)
        q_out = self.query.repeat(batch, 1, 1)
        q_out = q_out.view(batch, self.heads, self.channels // self.heads, 1)

        out = (q_out * k_out).sum(dim=2, keepdim=True) * self.temperature
        out = F.softmax(out, dim=-1)

        # print attention info
        if not self.training and x.get_device() == 1 and cfg.disp_attention:
            self.logger.info("Pooling:")
            for head in range(self.heads):
                self.logger.info("head {}".format(head))
                for h in range(height):
                    loggerInfo = "{:.3f} " * width
                    self.logger.info(loggerInfo.format(*out[0][head][0][h * width:(h + 1) * width].tolist()))

        out = (out * v_out).sum(dim=-1)
        out = out.view(batch, self.channels, 1, 1)

        return out

    def reset_parameters(self):
        # init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        # init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')

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


class SABasic(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, kernel_size,
                 heads=8, with_conv=False, r_dim=256, logger=None):
        super(SABasic, self).__init__()
        self.stride = stride
        self.heads = heads
        self.kernel_size = kernel_size
        self.with_conv = with_conv

        padding = get_same_padding(kernel_size)
        self.sa_conv_1 = SAConv(in_channels, out_channels, kernel_size, stride, padding, heads, r_dim=r_dim,
                                logger=logger)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.sa_conv_2 = SAConv(out_channels, out_channels, kernel_size, 1, padding, heads, r_dim=r_dim, logger=logger)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if with_conv:
            self.conv_1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

            self.conv_2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x, r=None):
        if self.with_conv:
            if self.training:
                shake_config = (True, True, True)
            else:
                shake_config = (False, False, False)
            alpha1, beta1 = get_alpha_beta(x.size(0), shake_config, x.device)
            sa_out_1 = self.sa_conv_1(x, r)
            sa_out_1 = self.bn1(sa_out_1)
            sa_out_1 = F.relu(sa_out_1)
            conv_out_1 = self.conv1(x)
            out = shake_function(sa_out_1, conv_out_1, alpha1, beta1)
            sa_out_2 = self.sa_conv_2(out, r)
            sa_out_2 = self.non_linear_2(sa_out_2)
            conv_out_2 = self.conv_2(out)
            alpha2, beta2 = get_alpha_beta(x.size(0), shake_config, x.device)
            out = shake_function(sa_out_2, conv_out_2, alpha2, beta2)
        else:
            out = self.sa_conv_1(x, r)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.sa_conv_2(out, r)
            out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class SABottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, groups=1, expansion=4,
                 base_width=64, heads=8, r_dim=256, logger=None):
        super(SABottleneck, self).__init__()
        self.stride = stride
        self.heads = heads
        self.kernel_size = kernel_size
        self.with_conv = cfg.model.with_conv
        self.expansion = expansion
        self.rezero = cfg.model.rezero
        if self.rezero:
            self.scale = nn.Parameter(torch.Tensor([0]))

        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )

        padding = get_same_padding(kernel_size)
        self.sa_conv = SAConv(width, width, kernel_size, stride, padding, heads, r_dim=r_dim, logger=logger)
        self.non_linear = nn.Sequential(
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )

        if self.with_conv:
            self.conv_2_2 = nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, stride=self.stride, padding=1, bias=False),
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

    def conv_2_1(self, out, r):
        out = self.sa_conv(out, r)
        return self.non_linear(out)

    def forward(self, x, r=None):
        out = self.conv1(x)

        if self.with_conv:
            if self.training:
                shake_config = (True, True, True)
            else:
                shake_config = (False, False, False)
            alpha, beta = get_alpha_beta(x.size(0), shake_config, x.device)
            out1 = self.conv_2_1(out, r)
            out2 = self.conv_2_2(out)
            out = shake_function(out1, out2, alpha, beta)
        else:
            out = self.conv_2_1(out, r)

        out = self.conv3(out)

        if self.rezero:
            out = out * self.scale + self.shortcut(x)
        else:
            out += self.shortcut(x)
        out = F.relu(out)

        return out


class PoolBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, groups=1, expansion=4, base_width=64, **kwargs):
        super(PoolBottleneck, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.with_conv = cfg.model.with_conv
        self.expansion = expansion
        self.rezero = cfg.model.rezero
        if self.rezero:
            self.scale = nn.Parameter(torch.Tensor([0]))

        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            MixedConv2d(width, width, [3, 5, 7, 9], stride, depthwise=True),
            # nn.AvgPool2d(kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            ChannelShuffle(4),
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

    def forward(self, x, r=None):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.rezero:
            out = out * self.scale + self.shortcut(x)
        else:
            out += self.shortcut(x)

        out = F.relu(out)
        return out
