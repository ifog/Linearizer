"""Image-classification data loaders.

Supports MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100. All datasets are loaded
from torchvision, cached under /home/nvidia/data (existing) or /tmp/claude/data.

The `subset_size` knob lets us run quick CPU/GPU smoke tests on a small slice
of the training set without changing any other code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_ROOT = "/home/nvidia/data"


@dataclass
class VisionLoaders:
    train: DataLoader
    test: DataLoader
    n_classes: int
    in_channels: int
    img_size: int
    name: str


_MEAN_STD = {
    "mnist":         ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10":       ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100":      ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
}

_SPECS = {
    "mnist":         (datasets.MNIST,        10,  1, 28),
    "fashion_mnist": (datasets.FashionMNIST, 10,  1, 28),
    "cifar10":       (datasets.CIFAR10,      10,  3, 32),
    "cifar100":      (datasets.CIFAR100,    100,  3, 32),
}


def _transforms(name: str, train: bool) -> transforms.Compose:
    mean, std = _MEAN_STD[name]
    aug: list = []
    if train and name in ("cifar10", "cifar100"):
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    return transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def get_loaders(
    name: str,
    *,
    batch_size: int = 128,
    test_batch_size: Optional[int] = None,
    subset_size: Optional[int] = None,
    num_workers: int = 2,
    root: str = DATA_ROOT,
    seed: int = 0,
) -> VisionLoaders:
    """Build train/test DataLoaders for one of the supported datasets.

    `subset_size`, if given, restricts the *training* set to that many
    randomly-chosen samples (a fixed subset by `seed`). Used for fast pilot
    runs. Test set is always the full benchmark test set.
    """
    if name not in _SPECS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(_SPECS)}")
    ds_class, n_classes, in_channels, img_size = _SPECS[name]
    test_batch_size = test_batch_size or 2 * batch_size

    train_ds = ds_class(root, train=True,  download=True, transform=_transforms(name, train=True))
    test_ds  = ds_class(root, train=False, download=True, transform=_transforms(name, train=False))

    if subset_size is not None and subset_size < len(train_ds):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train_ds), generator=g)[:subset_size]
        train_ds = Subset(train_ds, idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return VisionLoaders(
        train=train_loader, test=test_loader,
        n_classes=n_classes, in_channels=in_channels, img_size=img_size, name=name,
    )
