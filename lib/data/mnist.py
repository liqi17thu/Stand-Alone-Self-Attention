import torch
from torchvision import datasets, transforms
from lib.config import cfg


def mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.workers
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.workers
    )

    return train_loader, test_loader, 10
