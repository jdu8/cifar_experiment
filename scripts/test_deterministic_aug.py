"""
Verify that DeterministicAugDataset produces identical augmented images
for the same (epoch, sample_idx) across multiple passes, and different
images across epochs.

Run: python scripts/test_deterministic_aug.py
"""

import sys
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DeterministicAugDataset(Dataset):
    """
    Wraps a base dataset and applies transform deterministically per (epoch, idx).
    Setting the same epoch before two passes guarantees identical augmented images.
    """
    def __init__(self, base_dataset, transform):
        self.base      = base_dataset
        self.transform = transform
        self.epoch     = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        seed = self.epoch * len(self) + idx
        torch.manual_seed(seed)
        random.seed(seed)
        img = self.transform(img)
        return img, label, idx


def collect_pass(loader):
    """Collect {idx: tensor} from one full pass through loader."""
    result = {}
    for imgs, labels, indices in loader:
        for i, idx in enumerate(indices.tolist()):
            result[idx] = imgs[i].clone()
    return result


def check_identical(pass_a, pass_b, label_a="pass A", label_b="pass B"):
    n_diff = sum(
        not torch.equal(pass_a[i], pass_b[i])
        for i in pass_a
    )
    if n_diff == 0:
        print(f"  PASS — {label_a} and {label_b} are identical ({len(pass_a)} samples)")
        return True
    else:
        print(f"  FAIL — {n_diff}/{len(pass_a)} samples differ between {label_a} and {label_b}")
        return False


def check_different(pass_a, pass_b, label_a="epoch A", label_b="epoch B"):
    n_same = sum(
        torch.equal(pass_a[i], pass_b[i])
        for i in pass_a
    )
    if n_same < len(pass_a):
        print(f"  PASS — {len(pass_a) - n_same}/{len(pass_a)} samples differ between {label_a} and {label_b} (expected)")
        return True
    else:
        print(f"  FAIL — all samples are identical across {label_a} and {label_b} (seeding not working)")
        return False


def run_tests(data_dir='data', n_samples=200, num_workers=0):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    base = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    subset = Subset(base, list(range(n_samples)))
    aug_ds = DeterministicAugDataset(subset, transform_train)

    loader_seq  = DataLoader(aug_ds, batch_size=64, shuffle=False, num_workers=num_workers)
    loader_shuf = DataLoader(aug_ds, batch_size=64, shuffle=True,  num_workers=num_workers,
                             generator=torch.Generator().manual_seed(0))

    all_passed = True

    # ── Test 1: same epoch, sequential loader, two passes ──────────────────
    print(f"\nTest 1 (num_workers={num_workers}): same epoch, sequential, two passes")
    aug_ds.set_epoch(5)
    p1 = collect_pass(loader_seq)
    p2 = collect_pass(loader_seq)
    all_passed &= check_identical(p1, p2, "pass 1", "pass 2")

    # ── Test 2: same epoch, shuffled vs sequential ──────────────────────────
    print(f"Test 2 (num_workers={num_workers}): same epoch, shuffled vs sequential")
    aug_ds.set_epoch(5)
    p3 = collect_pass(loader_shuf)
    all_passed &= check_identical(p1, p3, "sequential", "shuffled")

    # ── Test 3: different epochs produce different augmentations ────────────
    print(f"Test 3 (num_workers={num_workers}): different epochs differ")
    aug_ds.set_epoch(6)
    p4 = collect_pass(loader_seq)
    all_passed &= check_different(p1, p4, "epoch 5", "epoch 6")

    # ── Test 4: spot-check a single sample visually ─────────────────────────
    print(f"Test 4 (num_workers={num_workers}): spot-check sample 0 pixel values")
    aug_ds.set_epoch(5)
    img_a, _, _ = aug_ds[0]
    aug_ds.set_epoch(5)
    img_b, _, _ = aug_ds[0]
    aug_ds.set_epoch(6)
    img_c, _, _ = aug_ds[0]

    same_epoch = torch.equal(img_a, img_b)
    diff_epoch = not torch.equal(img_a, img_c)
    if same_epoch and diff_epoch:
        print(f"  PASS — sample 0 is identical within epoch 5, different in epoch 6")
        print(f"         epoch5 pixel[0,0]: {img_a[0,0,:3].numpy()}")
        print(f"         epoch6 pixel[0,0]: {img_c[0,0,:3].numpy()}")
    else:
        print(f"  FAIL — same_epoch={same_epoch}, diff_epoch={diff_epoch}")
        all_passed = False

    return all_passed


def main():
    print("=" * 60)
    print("Deterministic augmentation verification")
    print("=" * 60)

    passed_0  = run_tests(num_workers=0)

    print("\n" + "=" * 60)
    if passed_0:
        print("All tests PASSED with num_workers=0")
    else:
        print("Some tests FAILED — check output above")
    print("=" * 60)


if __name__ == '__main__':
    main()
