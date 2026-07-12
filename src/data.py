from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as T


def get_transforms(image_size: int = 224, is_train: bool = True):
    """
    ViT requires 224x224 input (pretrained on ImageNet at this resolution).
    CIFAR-10 is 32x32 so we upscale + augment for training.
    """
    if is_train:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(image_size, padding=image_size // 8),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )


def build_dataloaders(cfg: dict):
    if not Path(cfg["data_dir"]).exists():
        train_dataset = CIFAR10(
        root=cfg["data_dir"],
        train=True,
        download=True,
        transform=get_transforms(cfg["image_size"], is_train=True),
        )
        test_dataset = CIFAR10(
            root=cfg["data_dir"],
            train=False,
            download=True,
            transform=get_transforms(cfg["image_size"], is_train=False),
        )
    else:
        train_dataset = CIFAR10(
            root=cfg["data_dir"],
            train=True,
            download=False,
            transform=get_transforms(cfg["image_size"], is_train=True),
        )
        test_dataset = CIFAR10(
            root=cfg["data_dir"],
            train=False,
            download=False,
            transform=get_transforms(cfg["image_size"], is_train=False),
        )

    n_val = 5000
    n_train = len(train_dataset) - n_val
    train_ds, val_ds = random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    kw = dict(
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True
    )
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_dataset, shuffle=False, **kw)

    print(
        f"Train : {len(train_ds):,} | Val : {len(val_ds):,} | Test : {len(test_dataset):,}"
    )
    return train_loader, val_loader, test_loader
