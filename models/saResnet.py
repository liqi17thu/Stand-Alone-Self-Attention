import torch.nn as nn
import torch.nn.functional as F

from models.units.saUnit import SAStem, SABottleneck
from models.units.resUnit import Bottleneck

class SAResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, stem=False):
        super(SAResNet, self).__init__()
        self.in_places = 64

        if stem:
            self.init = nn.Sequential(
                # CIFAR10
                SAStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # For ImageNet
                # AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(4, 4)
            )
        else:
            self.init = nn.Sequential(
                # CIFAR10
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # For ImageNet
                # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        if isinstance(block, list):
            self.layer1 = self._make_layer(block[0], 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block[1], 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block[2], 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block[3], 512, num_blocks[3], stride=2)
            self.dense = nn.Linear(512 * block[3].expansion, num_classes)
        else:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.dense = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_places, planes, stride))                    
            self.in_places = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out


def SAResNet26(num_classes=1000, stem=False, num_sablock=2):
    block = [Bottleneck for _ in range(4 - num_sablock)] + [SABottleneck for _ in range(num_sablock)]
    return SAResNet(block, [1, 2, 4, 1], num_classes=num_classes, stem=stem)


def SAResNet38(num_classes=1000, stem=False, num_sablock=2):
    block = [Bottleneck for _ in range(4 - num_sablock)] + [SABottleneck for _ in range(num_sablock)]
    return SAResNet(SABottleneck, [2, 3, 5, 2], num_classes=num_classes, stem=stem)


def SAResNet50(num_classes=1000, stem=False, num_sablock=2):
    block = [Bottleneck for _ in range(4 - num_sablock)] + [SABottleneck for _ in range(num_sablock)]
    return SAResNet(SABottleneck, [3, 4, 6, 3], num_classes=num_classes, stem=stem)


# temp = torch.randn((2, 3, 224, 224))
# model = ResNet38(num_classes=1000, stem=True)
# print(get_model_parameters(model))