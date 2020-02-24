import torch.nn as nn

from .activation import Hsigmoid
from lib.utils import make_divisible

class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION

        num_mid = make_divisible(self.channel // self.reduction, divisor=8)

        self.fc = nn.Sequential(
            nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True),
            Hsigmoid(inplace=True),
        )

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y