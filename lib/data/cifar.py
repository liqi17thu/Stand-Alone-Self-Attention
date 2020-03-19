import torch
from torchvision import datasets, transforms

from lib.config import cfg
from lib.data.data_util import CIFAR10Policy

def cifar10():
    if cfg.ddp.local_rank == 0:
        print('Load Dataset :: {}'.format(cfg.dataset.name))

    if cfg.dataset.use_aa:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.dataset.mean,
                std=cfg.dataset.std
            )
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.dataset.mean,
                std=cfg.dataset.std
            )
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.dataset.mean,
            std=cfg.dataset.std
        )
    ])

    train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('data', train=False, transform=transform_test)

    if cfg.ddp.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = None
        test_sampler = None

    if cfg.ddp.distributed:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cfg.dataset.batch_size,
            sampler=train_sampler,
            num_workers=cfg.dataset.workers
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=cfg.dataset.batch_size,
            sampler=test_sampler,
            num_workers=cfg.dataset.workers
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.workers
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.workers
        )

    return [train_loader, test_loader], [train_sampler, test_sampler], 10


def cifar100(cfg):

    if cfg.dataset.use_aa:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.dataset.mean,
                std=cfg.dataset.std
            )
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.dataset.mean,
                std=cfg.dataset.std
            )
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.dataset.mean,
            std=cfg.dataset.mean
        ),
    ])

    train_data = datasets.CIFAR100('data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR100('data', train=False, transform=transform_test)

    if cfg.ddp.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = None
        test_sampler = None

    if cfg.ddp.distributed:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cfg.dataset.batch_size,
            sampler=train_sampler,
            num_workers=cfg.dataset.workers
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=cfg.dataset.batch_size,
            sampler=test_sampler,
            num_workers=cfg.dataset.workers
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.workers
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.workers
        )

    return [train_loader, test_loader], [train_sampler, test_sampler], 100
