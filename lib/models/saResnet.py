import torch.nn as nn
import torch.nn.functional as F

from .units.saUnit import SAStem, SABottleneck
from .units.resUnit import Bottleneck


class SAResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv'):
        super(SAResNet, self).__init__()
        self.in_places = 64
        self.heads = heads
        self.kernel_size = kernel_size

        if stem.split('_')[1] == 'sa':
            if stem.split('_')[0] == 'cifar':
                self.init = nn.Sequential(
                    SAStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            else:
                self.init = nn.Sequential(
                    SAStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(4, 4)
                )
        else:
            if stem.split('_')[0] == 'cifar':
                self.init = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            else:
                self.init = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        self.layer1 = self._make_layer(block[0], 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block[1], 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block[3], 512, num_blocks[3], stride=2)
        self.dense = nn.Linear(512 * block[3].expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if block.__name__ == "SABottleneck":
                layers.append(block(self.in_places, planes, stride, kernel_size=self.kernel_size, heads=self.heads))
            else:
                layers.append(block(self.in_places, planes, stride))
            self.in_places = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out


def SAResNet26(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_sablock=2):
    block = [Bottleneck for _ in range(4 - num_sablock)] + [SABottleneck for _ in range(num_sablock)]
    return SAResNet(block, [1, 2, 4, 1], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem)


def SAResNet38(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_sablock=2):
    block = [Bottleneck for _ in range(4 - num_sablock)] + [SABottleneck for _ in range(num_sablock)]
    return SAResNet(block, [2, 3, 5, 2], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem)


def SAResNet50(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_sablock=2):
    block = [Bottleneck for _ in range(4 - num_sablock)] + [SABottleneck for _ in range(num_sablock)]
    return SAResNet(block, [3, 4, 6, 3], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem)


def ResNet101(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_sablock=2):
    block = [Bottleneck for _ in range(4 - num_sablock)] + [SABottleneck for _ in range(num_sablock)]
    return SAResNet(block, [3, 4, 23, 3], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem)


def ResNet152(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_sablock=2):
    block = [Bottleneck for _ in range(4 - num_sablock)] + [SABottleneck for _ in range(num_sablock)]
    return SAResNet(block, [3, 4, 36, 3], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem)