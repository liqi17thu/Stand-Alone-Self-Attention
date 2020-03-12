import torch
import torch.nn as nn
import torch.nn.init as init

from .units.saUnit import SAStem, SABasic, SABottleneck
from .units.resUnit import Bottleneck, BasicBlock

from lib.config import cfg


class SAResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, heads, kernel_size, stem, num_resblock, logger):
        super(SAResNet, self).__init__()
        self.in_places = 40
        self.heads = heads
        self.kernel_size = kernel_size
        self.num_resblock = num_resblock
        self.logger = logger
        self.r_dim = cfg.model.r_dim
        self.encoding = cfg.model.encoding

        if stem.split('_')[1] == 'sa':
            if stem.split('_')[0] == 'cifar':
                self.init = nn.Sequential(
                    SAStem(in_channels=3, out_channels=40, kernel_size=4, stride=1, padding=2, groups=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            else:
                self.init = nn.Sequential(
                    SAStem(in_channels=3, out_channels=40, kernel_size=4, stride=1, padding=2, groups=1),
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

        self.layer1 = self._make_layer(block[0], 40, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block[1], 80, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block[2], 112, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block[3], 160, num_blocks[3], stride=2)
        self.layers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)
        self.dense = nn.Linear(512 * block[3].expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if cfg.model.encoding == "xl":
            self.r = nn.Parameter(torch.randn(1, self.r_dim, self.kernel_size, self.kernel_size), requires_grad=True)
        else:
            self.r = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.encoding == "xl":
            init.normal_(self.r, 0, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if block.__name__ == "SABottleneck" or block.__name__ == "SABasic":
                layers.append(block(self.in_places, planes, stride, self.kernel_size, r_dim=self.r_dim,
                                    heads=self.heads, logger=self.logger))
            else:
                layers.append(block(self.in_places, planes, stride))
            self.in_places = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        for i in range(self.num_resblock):
            out = self.layers[i](out)
        for i in range(self.num_resblock, 4):
            if not self.training and x.get_device() == 1 and cfg.disp_attention:
                self.logger.info(f"layer {i+1}")
            for (j, layer) in enumerate(self.layers[i]):
                if not self.training and x.get_device() == 1 and cfg.disp_attention:
                    self.logger.info(f"block {j}")
                out = layer(out, self.r)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out


def SAResNet17(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, attention_logger=None):
    block = [Bottleneck for _ in range(num_resblock)] + [SABottleneck for _ in range(4 - num_resblock)]
    return SAResNet(block, [1, 2, 1, 1], num_classes, heads, kernel_size, stem, num_resblock, attention_logger)


def SAResNet18(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, attention_logger=None):
    block = [BasicBlock for _ in range(num_resblock)] + [SABasic for _ in range(4 - num_resblock)]
    return SAResNet(block, [2, 2, 2, 2], num_classes, heads, kernel_size, stem, num_resblock, attention_logger)


def SAResNet20(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, attention_logger=None):
    block = [Bottleneck for _ in range(num_resblock)] + [SABottleneck for _ in range(4 - num_resblock)]
    return SAResNet(block, [1, 2, 2, 1], num_classes, heads, kernel_size, stem, num_resblock, attention_logger)


def SAResNet26(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, attention_logger=None):
    block = [Bottleneck for _ in range(num_resblock)] + [SABottleneck for _ in range(4 - num_resblock)]
    return SAResNet(block, [1, 2, 4, 1], num_classes, heads, kernel_size, stem, num_resblock, attention_logger)


def SAResNet38(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, attention_logger=None):
    block = [Bottleneck for _ in range(num_resblock)] + [SABottleneck for _ in range(4 - num_resblock)]
    return SAResNet(block, [2, 3, 5, 2], num_classes, heads, kernel_size, stem, num_resblock, attention_logger)


def SAResNet50(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, attention_logger=None):
    block = [Bottleneck for _ in range(num_resblock)] + [SABottleneck for _ in range(4 - num_resblock)]
    return SAResNet(block, [3, 4, 6, 3], num_classes, heads, kernel_size, stem, num_resblock, attention_logger)


def SAResNet101(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, attention_logger=None):
    block = [Bottleneck for _ in range(num_resblock)] + [SABottleneck for _ in range(4 - num_resblock)]
    return SAResNet(block, [3, 4, 23, 3], num_classes, heads, kernel_size, stem, num_resblock, attention_logger)


def SAResNet152(num_classes=1000, heads=8, kernel_size=7, stem='cifar_conv', num_resblock=2, attention_logger=None):
    block = [Bottleneck for _ in range(num_resblock)] + [SABottleneck for _ in range(4 - num_resblock)]
    return SAResNet(block, [3, 4, 36, 3], num_classes, heads, kernel_size, stem, num_resblock, attention_logger)
