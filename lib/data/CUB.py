import json
import os

import torch
from PIL import Image
from torchvision import transforms

from lib.config import cfg


def identity(x):
    return x


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


def get_loader(file_list, batch_size, train=True):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.257, 0.276])
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg.TRAIN.DATASET.IMAGE_SIZE),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        size = int(cfg.TRAIN.DATASET.TEST_SIZE * 1.15)
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(cfg.TRAIN.DATASET.TEST_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = SimpleDataset(file_list, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                         num_workers=cfg.TRAIN.DATASET.WORKERS, pin_memory=True)
    return loader


def CUB(cfg):
    train_loader = get_loader(cfg.TRAIN.DATASET.TRAIN_DIR, cfg.TRAIN.DATASET.BATCH_SIZE, True)
    test_loader = get_loader(cfg.TRAIN.DATASET.TEST_DIR, cfg.TRAIN.DATASET.BATCH_SIZE, True)
    return train_loader, test_loader, 200
