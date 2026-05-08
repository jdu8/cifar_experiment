
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return the sample index."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, label, idx


def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_eval


def get_dataloaders(data_dir, val_size=5000, batch_size=256,
                    num_workers=2, seed=0, smoke_test=False,
                    eval_batch_size=None):
    transform_train, transform_eval = get_transforms()

    full_train_aug  = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True,
        transform=transform_train)
    full_train_eval = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True,
        transform=transform_eval)
    test_dataset    = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=transform_eval)

    rng     = np.random.RandomState(42)  # fixed split across all seeds
    targets = np.array(full_train_aug.targets)

    train_indices, val_indices = [], []
    val_per_class = val_size // 10

    for cls in range(10):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        val_indices.extend(cls_idx[:val_per_class].tolist())
        train_indices.extend(cls_idx[val_per_class:].tolist())

    if smoke_test:
        train_indices = train_indices[:1000]
        val_indices   = val_indices[:200]
        test_indices  = list(range(200))
    else:
        test_indices  = list(range(len(test_dataset)))

    # Standard train loader — no indices, shuffled, augmented
    train_loader = DataLoader(
        Subset(full_train_aug, train_indices),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0))

    # Indexed train loader — returns (data, label, index), shuffled, augmented
    # Used by train_reweight.py so weights are applied to correct samples
    train_loader_indexed = DataLoader(
        IndexedDataset(Subset(full_train_aug, train_indices)),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0))

    # Eval loaders — num_workers=0 avoids Windows spawn overhead (~10s per
    # loader); larger batch size compensates by reducing loop iterations and
    # improving GPU utilization.
    ebs = eval_batch_size or batch_size * 2
    train_eval_loader = DataLoader(
        Subset(full_train_eval, train_indices),
        batch_size=ebs, shuffle=False,
        num_workers=0, pin_memory=True)

    val_loader = DataLoader(
        Subset(full_train_eval, val_indices),
        batch_size=ebs, shuffle=False,
        num_workers=0, pin_memory=True)

    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=ebs, shuffle=False,
        num_workers=0, pin_memory=True)

    split_info = {
        'train_indices': train_indices,
        'val_indices':   val_indices,
        'test_indices':  test_indices,
        'n_train':       len(train_indices),
        'n_val':         len(val_indices),
        'n_test':        len(test_indices),
    }

    print(f"Split — Train: {len(train_indices)}, "
          f"Val: {len(val_indices)}, Test: {len(test_indices)}")

    return (train_loader, train_loader_indexed,
            train_eval_loader, val_loader, test_loader, split_info)
