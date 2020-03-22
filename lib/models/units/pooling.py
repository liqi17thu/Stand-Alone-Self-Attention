import torch
import torch.nn as nn


class AvgMaxPool2d(nn.Module):
    def __init__(self, kernel, stride, padding):
        super(AvgMaxPool2d, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([0]))

        self.avg = nn.AvgPool2d(kernel, stride=stride, padding=padding)
        self.max = nn.MaxPool2d(kernel, stride=stride, padding=padding)

    def forward(self, x):
        return self.avg(x) + self.max(x) * self.scale


def build_pooling_op(type, width, kernel, padding, stride):
    if type == 'avg':
        return nn.AvgPool2d(kernel, padding=padding, stride=stride)
    elif type == 'max':
        return nn.MaxPool2d(kernel, padding=padding, stride=stride)
    elif type == 'avgmax':
        return AvgMaxPool2d(kernel, padding=padding, stride=stride)
    elif type == 'spconv':
        return nn.Conv2d(width, width, kernel, padding=padding, stride=stride, groups=width)
    else:
        raise NotImplementedError
