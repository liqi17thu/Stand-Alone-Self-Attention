def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(d_model, max_len)
        position = torch.arange(0, max_len, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0::2, :] = torch.sin(div_term * position)
        pe[1::2, :] = torch.cos(div_term * position)
        pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch, channels, height, width = x.shape()
        x = x.view(batch, channels, -1)
        x = x + self.pe[0, :, :height * width]
        x = x.view(batch, channels, height, width)
        return self.dropout(x)