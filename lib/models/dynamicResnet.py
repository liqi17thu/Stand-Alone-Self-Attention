import torch.nn as nn

from .units.dynamicUnit import DynamicBottleneck
from .units.saUnit import SAStem
from .units.resUnit import Bottleneck


class DynamicResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', with_conv=False):
        super(DynamicResNet, self).__init__()
        self.in_places = 64
        self.heads = heads
        self.kernel_size = kernel_size
        self.with_conv = with_conv

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
            if block.__name__ == "DynamicBottleneck":
                layers.append(block(self.in_places, planes, stride, kernel_size=self.kernel_size, heads=self.heads, with_conv=self.with_conv))
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


def DynamicResNet26(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, with_conv=False):
    block = [Bottleneck for _ in range(num_resblock)] + [DynamicBottleneck for _ in range(4 - num_resblock)]
    return DynamicResNet(block, [1, 2, 4, 1], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem, with_conv=with_conv)


def DynamicResNet38(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, with_conv=False):
    block = [Bottleneck for _ in range(num_resblock)] + [DynamicBottleneck for _ in range(4 - num_resblock)]
    return DynamicResNet(block, [2, 3, 5, 2], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem, with_conv=with_conv)


def DynamicResNet50(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, with_conv=False):
    block = [Bottleneck for _ in range(num_resblock)] + [DynamicBottleneck for _ in range(4 - num_resblock)]
    return DynamicResNet(block, [3, 4, 6, 3], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem, with_conv=with_conv)


def DynamicResNet101(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, with_conv=False):
    block = [Bottleneck for _ in range(num_resblock)] + [DynamicBottleneck for _ in range(4 - num_resblock)]
    return DynamicResNet(block, [3, 4, 23, 3], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem, with_conv=with_conv)


def DynamicResNet152(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, with_conv=False):
    block = [Bottleneck for _ in range(num_resblock)] + [DynamicBottleneck for _ in range(4 - num_resblock)]
    return DynamicResNet(block, [3, 4, 36, 3], num_classes=num_classes, heads=heads, kernel_size=kernel_size, stem=stem, with_conv=with_conv)