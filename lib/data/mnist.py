import torch
from torchvision import datasets, transforms

def mnist(cfg):
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

    return train_loader, test_loader, 10
