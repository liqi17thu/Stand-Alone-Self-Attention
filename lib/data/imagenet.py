import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lib.data.data_util import ImageNetPolicy, ToBGRTensor


def imagenet(cfg):
    normalize = transforms.Normalize(mean=cfg.TRAIN.DATASET.MEAN, std=cfg.TRAIN.DATASET.STD)
    if cfg.TRAIN.DATASET.USE_AA:
        train_data = datasets.ImageFolder(
            cfg.TRAIN.DATASET.TRAIN_DIR,
            transforms.Compose([
                transforms.RandomResizedCrop(cfg.TRAIN.DATASET.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                ToBGRTensor() if cfg.TRAIN.DATASET.BGR else transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_data = datasets.ImageFolder(
            cfg.TRAIN.DATASET.TRAIN_DIR,
            transforms.Compose([
                transforms.RandomResizedCrop(cfg.TRAIN.DATASET.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4),
                ToBGRTensor() if cfg.TRAIN.DATASET.BGR else transforms.ToTensor(),
                normalize,
            ]))

    test_data = datasets.ImageFolder(
        cfg.TRAIN.DATASET.TEST_DIR,
        transforms.Compose([
            transforms.Resize(cfg.TRAIN.DATASET.TEST_RESIZE),
            transforms.CenterCrop(cfg.TRAIN.DATASET.TEST_SIZE),
            ToBGRTensor() if cfg.TRAIN.DATASET.BGR else transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.TRAIN.DATASET.BATCH_SIZE,
        shuffle=True,
        pin_memory=True, num_workers=cfg.TRAIN.DATASET.WORKERS)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.TRAIN.DATASET.BATCH_SIZE,
        shuffle=False,
        pin_memory=True, num_workers=cfg.TRAIN.DATASET.WORKERS)

    return train_loader, test_loader, 1000

