import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lib.data.data_util import ImageNetPolicy, ToBGRTensor
from lib.config import cfg

from .transformer_v2 import get_transforms


def finetune_imagenet():
    transformation = get_transforms(input_size=cfg.dataset.funetune_size, test_size=cfg.dataset.finetune_size,
                                    kind='full', crop=True, need=('train', 'val'), backbone=None)
    transform_test = transformation['val']

    train_data = datasets.ImageFolder(
        cfg.dataset.train_dir,
        transform_test
    )

    test_data = datasets.ImageFolder(
        cfg.dataset.test_dir,
        transform_test
    )

    if cfg.ddp.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = None
        test_sampler = None

    if cfg.ddp.distributed:
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=cfg.dataset.batch_size,
            sampler=train_sampler,
            pin_memory=True, num_workers=cfg.dataset.workers)

        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=cfg.dataset.batch_size,
            sampler=test_sampler,
            pin_memory=True, num_workers=cfg.dataset.workers)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=cfg.dataset.batch_size,
            shuffle=True,
            pin_memory=True, num_workers=cfg.dataset.workers)

        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=cfg.dataset.batch_size,
            shuffle=False,
            pin_memory=True, num_workers=cfg.dataset.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler], 1000
