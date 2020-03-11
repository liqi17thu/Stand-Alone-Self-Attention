import torch
import tarfile
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from lib.data.data_util import ImageNetPolicy, SubsetDistributedSampler, ToBGRTensor
from lib.config import cfg


class ImageTarDataset(Dataset):

    def __init__(self, tar_file, transform=None):
        # time0 = pc()
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        with tarfile.open(tar_file, 'r:') as tar:
            # time1 = pc()
            for tar_member in tar.getmembers():
                if tar_member.name.count('/') != 2:
                    continue

                categories_set.add(self.get_category_from_filename(tar_member.name))
                self.tar_members.append(tar_member)
            # time2 = pc()
        index = 0
        categories_set = sorted(categories_set)
        for category in categories_set:
            self.categories[category] = index
            index += 1
        self.transform = transform
        # print("tarfile.open: ", time1 - time0, " categorization:", time2-time1, " sorting: ", pc() - time2)

    def get_category_from_filename(self, filename):
        begin = filename.find('/')
        begin += 1
        end = filename.find('/', begin)
        return filename[begin:end]

    def __len__(self):
        return len(self.tar_members)

    def __getitem__(self, index):
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, 'r:')

        sample = self.tar_handle.extractfile(self.tar_members[index])
        sample = Image.open(sample)
        sample = sample.convert('RGB')
        if self.transform:
            sample = self.transform(sample)
        category = self.categories[self.get_category_from_filename(self.tar_members[index].name)]
        return sample, category


data_loader = ImageTarDataset


def imagenet_tar():
    normalize = transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
    if cfg.dataset.use_aa:
        train_data = data_loader(
            cfg.dataset.train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(cfg.dataset.image_size),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                ToBGRTensor() if cfg.dataset.bgr else transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_data = data_loader(
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

    test_data = data_loader(
        cfg.dataset.test_dir,
        transforms.Compose([
            transforms.Resize(cfg.dataset.test_resize),
            transforms.CenterCrop(cfg.dataset.test_size),
            ToBGRTensor() if cfg.dataset.bgr else transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.dataset.batch_size,
        shuffle=True,
        pin_memory=True, num_workers=cfg.dataset.workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.dataset.batch_size,
        shuffle=False,
        pin_memory=True, num_workers=cfg.dataset.workers)

    return train_loader, test_loader, 1000