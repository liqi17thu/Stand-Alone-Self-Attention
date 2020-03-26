import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lib.data.data_util import ImageNetPolicy, ToBGRTensor
from lib.config import cfg
from lib.data.transformer_v2 import Resize, CenterCrop

def imagenet():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
    if cfg.dataset.use_aa:
        train_data = datasets.ImageFolder(
            cfg.dataset.train_dir,
            transforms.Compose(
                [Resize(int((256 / 224) * cfg.dataset.finetune_size)),  # to maintain same ratio w.r.t. 224 images
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(0.05, 0.05, 0.05),
                 CenterCrop(cfg.dataset.finetune_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)
            ])
        )
    else:
        train_data = datasets.ImageFolder(
            cfg.dataset.train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(cfg.dataset.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4),
                ToBGRTensor() if cfg.dataset.bgr else transforms.ToTensor(),
                normalize,
            ]))

    test_data = datasets.ImageFolder(
        cfg.dataset.test_dir,
        transforms.Compose([
           transforms.Resize(cfg.dataset.test_resize),
           transforms.CenterCrop(cfg.dataset.image_size),
           ToBGRTensor() if cfg.dataset.bgr else transforms.ToTensor(),
           normalize,
        ])
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

