'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import os
import numpy as np


from lib.config import cfg
from lib.core.train import train
from lib.core.vaild import validate
from lib.data import *
from lib.models.saMobilenet import *
from lib.models.saResnet import *
from lib.utils import CrossEntropyLabelSmooth, get_scheduler, get_net_info
from lib.utils import save_checkpoint, get_model_parameters, get_logger, get_attention_logger

if cfg.ddp.distributed:
    from torch.nn.parallel import DistributedDataParallel as DDP


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out

if cfg.ddp.distributed:
    # np.random.seed(cfg.ddp.seed)
    torch.manual_seed(cfg.ddp.seed)
    torch.cuda.manual_seed_all(cfg.ddp.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.cuda.set_device(cfg.ddp.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method=cfg.ddp.dist_url,
                                            world_size=cfg.ddp.gpus, rank=cfg.ddp.local_rank,
                                            group_name='mtorch')
    gpu_id = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)

model = MobileNetV3_Large()

if cfg.ddp.local_rank == 0:
    if cfg.test:
        logger = get_logger(os.path.join(cfg.save_path, 'test.log'))
    else:
        logger = get_logger(os.path.join(cfg.save_path, 'train.log'))
    attention_logger = get_attention_logger(os.path.join(cfg.save_path, 'attention.log'))
    writer = SummaryWriter(cfg.log_dir)
else:
    logger = None
    attention_logger = None
    writer = None

loaders, samplers, num_classes = eval(cfg.dataset.name)()
train_loader, test_loader = loaders
train_sampler, test_sampler = samplers

if cfg.ddp.local_rank == 0:
    print('Model Name: {0}'.format(cfg.model.name))

if cfg.crit.smooth > 0:
    criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=cfg.crit.smooth)
else:
    criterion = nn.CrossEntropyLoss()

if cfg.cuda:
    if cfg.ddp.distributed:
        model = model.to(torch.cuda.current_device())
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()
    criterion = criterion.cuda()

if cfg.model.pre_trained:
    checkpoint = torch.load('/philly/rr1/resrchvc_scratch/v-liqi/mobilenetv3/mbv3_large.old.pth.tar')

    # if cfg.ddp.local_rank == 0:
    #     import ipdb; ipdb.set_trace()

    state_dict = checkpoint['state_dict']

    keys = state_dict.copy().keys()
    for key in keys:
        if 'total_ops' in key or 'total_params' in key:
            del state_dict[key]

    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = cfg.train.start_epoch
else:
    start_epoch = cfg.train.start_epoch
    best_acc = 0.0

get_net_info(model, (3, cfg.dataset.image_size, cfg.dataset.image_size),
                logger=logger, local_rank=cfg.ddp.local_rank)

if cfg.optim.method == 'SGD':
    optimizer = optim.SGD(model.parameters(), **cfg.optim.sgd_params)
elif cfg.optim.method == 'Adam':
    optimizer = optim.Adam(model.parameters(), **cfg.optim.adam_params)
else:
    raise ValueError(f'Unsupported optimization: {cfg.optim.method}')

scheduler = get_scheduler(optimizer, len(train_loader), cfg)


_ = validate(model, test_loader, criterion, start_epoch, logger, attention_logger, writer)
