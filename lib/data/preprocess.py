import torch
from torchvision import datasets, transforms


def load_data(cfg):
    print('Load Dataset :: {}'.format(cfg.TRAIN.DATASET.NAME))
    if cfg.TRAIN.DATASET.NAME == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.TRAIN.DATASET.MEAN,
                std=cfg.TRAIN.DATASET.MEAN
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.TRAIN.DATASET.MEAN,
                std=cfg.TRAIN.DATASET.MEAN
            )
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
            batch_size=cfg.TRAIN.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.TRAIN.DATASET.WORKERS
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transform_test),
            batch_size=cfg.TRAIN.DATASET.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.DATASET.WORKERS
        )

    elif cfg.TRAIN.DATASET.NAME == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.TRAIN.DATASET.MEAN,
                std=cfg.TRAIN.DATASET.MEAN
            ),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.TRAIN.DATASET.MEAN,
                std=cfg.TRAIN.DATASET.MEAN
            ),
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True, transform=transform_train),
            batch_size=cfg.TRAIN.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.TRAIN.DATASET.WORKERS
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, transform=transform_test),
            batch_size=cfg.TRAIN.DATASET.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.DATASET.WORKERS
        )

    elif cfg.TRAIN.DATASET.NAME == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,)
            )
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=cfg.TRAIN.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.TRAIN.DATASET.WORKERS
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform),
            batch_size=cfg.TRAIN.DATASET.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.DATASET.WORKERS
        )

    elif cfg.TRAIN.DATASET.NAME == 'IMAGENET':
        pass

    return train_loader, test_loader
