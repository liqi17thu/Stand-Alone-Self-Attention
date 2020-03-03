import torch
import torch.nn as nn

import math


class PositionalEncoding(nn.Module):
    def __init__(self, out_channels, kernel_size, heads, bias=False, encoding='learnable', r_dim=256):
        super(PositionalEncoding, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.heads = heads
        self.encoding = encoding

        if self.encoding == "learnable":
            self.encoder = LearnablePostionalEncoding(out_channels, kernel_size)
        elif self.encoding == "sine":
            self.encoder = SinePositionalEncoding(out_channels // heads)
        elif self.encoding == "xl":
            self.encoder = XLPositionalEncoding(out_channels, heads, r_dim, bias)
        else:
            raise NotImplementedError

    def forward(self, k_out, r=None):
        # shape of K_out:
        # batch, channels, height, width, kernel, kernel
        if self.encoding == "learnable":
            return self.encoder(k_out)
        elif self.encoding == "sine":
            batch, channels, height, width, kernel, kernel = k_out.size()
            # batch, heads, out/heads, H/s, W/s, -1
            k_out = k_out.contiguous().view(batch, self.heads, channels // self.heads, height, width, -1)
            k_out = k_out.permute(0, 1, 3, 4, 2, 5).contiguous()
            k_out = k_out.view(-1, channels // self.heads, kernel * kernel)
            k_out = self.encoder(k_out)
            k_out = k_out.view(batch, self.heads, height, width, channels // self.heads, -1)
            k_out = k_out.permute(0, 1, 4, 2, 3, 5).contiguous()
            return k_out.view(batch, channels, height, width, kernel, kernel)
        elif self.encoding == "xl":
            r_out, u, v = self.encoder(r)
            return k_out, r_out, u, v
        else:
            raise NotImplementedError


class LearnablePostionalEncoding(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super(LearnablePostionalEncoding, self).__init__()
        self.out_channels = out_channels
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

    def forward(self, k_out):
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        return torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)


class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(SinePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(d_model, max_len)
        position = torch.arange(0, max_len, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).unsqueeze(1)
        pe[0::2, :] = torch.sin(div_term * position)
        pe[1::2, :] = torch.cos(div_term * position)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[0, :, :x.shape[2]]
        return self.dropout(x)


class XLPositionalEncoding(nn.Module):
    def __init__(self, out_channels, heads, r_dim, bias=False):
        super(XLPositionalEncoding, self).__init__()
        self.heads = heads

        self.u = nn.Parameter(torch.randn(1, heads, out_channels // heads, 1, 1, 1), requires_grad=True)
        self.v = nn.Parameter(torch.randn(1, heads, out_channels // heads, 1, 1, 1), requires_grad=True)
        self.key_position_conv = nn.Conv2d(r_dim, out_channels, kernel_size=1, bias=bias, groups=heads)

    def forward(self, r):
        r_out = self.key_position_conv(r)
        r_out = r_out.view(1, self.heads, self.out_channels // self.heads, 1, 1, -1)
        return r_out, self.u, self.v
