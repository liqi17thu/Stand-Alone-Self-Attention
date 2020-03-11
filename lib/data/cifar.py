import torch
from torchvision import datasets, transforms

from lib.config import cfg


def cifar10():
    print('Load Dataset :: {}'.format(cfg.TRAIN.DATASET.NAME))
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

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.workers
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform_test),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.workers
    )
    return train_loader, test_loader, 10


def cifar100(cfg):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.dataset.mean,
            std=cfg.dataset.std
        ),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.dataset.mean,
            std=cfg.dataset.mean
        ),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True, transform=transform_train),
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.workers
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=False, transform=transform_test),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.workers
    )
    return train_loader, test_loader, 100