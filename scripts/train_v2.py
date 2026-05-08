"""
train_v2.py — reweighting with geometrically aligned augmentations.

Key difference from train_reweight.py:
  Each epoch, train and embedding-eval use the SAME deterministic augmentation
  (seeded by epoch * n_train + sample_idx), so weights are computed in the
  same augmented embedding space that training actually operates in.

Loop structure:
  epoch 0  : train with uniform weights (random aug, establishes baseline embeddings)
  epoch 1+ : eval train with aug_e → compute weights → train with same aug_e
  every epoch: eval val + test (clean images, for fair comparison)
"""

import os
import warnings
import random

os.environ.setdefault('PYTHONWARNINGS', 'ignore::FutureWarning')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*align should be passed.*')

import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.model   import get_model
from utils.metrics import (evaluation_pass, compute_metrics_summary,
                           compute_neighbors_faiss,
                           compute_within_class_neighbors_faiss,
                           compute_purity, compute_displacement)
from utils.animation import (fit_umap_reducer, save_epoch_plots,
                             make_displacement_plot)
from utils.data import get_dataloaders

CIFAR100_SUPERCLASSES = {
    'aquatic_mammals':        [4, 30, 55, 72, 95],
    'fish':                   [1, 32, 67, 73, 91],
    'flowers':                [54, 62, 70, 82, 92],
    'food_containers':        [9, 10, 16, 28, 61],
    'fruit_vegetables':       [0, 51, 53, 57, 83],
    'household_electrical':   [22, 39, 40, 86, 87],
    'household_furniture':    [5, 20, 25, 84, 94],
    'insects':                [6, 7, 14, 18, 24],
    'large_carnivores':       [3, 42, 43, 88, 97],
    'large_outdoor_man_made': [12, 17, 37, 68, 76],
    'large_outdoor_natural':  [23, 33, 49, 60, 71],
    'large_omnivores_herbs':  [15, 19, 21, 31, 38],
    'medium_mammals':         [34, 63, 64, 66, 75],
    'invertebrates':          [26, 45, 77, 79, 99],
    'people':                 [2, 11, 35, 46, 98],
    'reptiles':               [27, 29, 44, 78, 93],
    'small_mammals':          [36, 50, 65, 74, 80],
    'trees':                  [47, 52, 56, 59, 96],
    'vehicles_1':             [8, 13, 48, 58, 90],
    'vehicles_2':             [41, 69, 81, 85, 89],
}


# ── Deterministic augmentation dataset ────────────────────────────────────────

class DeterministicAugDataset(Dataset):
    """
    Same (epoch, idx) → identical augmented image, guaranteed.
    Call set_epoch() before each pass to change the augmentation set.
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


def verify_determinism(aug_dataset, n_check=32):
    """
    Quick pre-training sanity check: two sequential passes must be identical.
    Raises RuntimeError if they differ.
    """
    loader = DataLoader(aug_dataset, batch_size=n_check, shuffle=False,
                        num_workers=0)
    aug_dataset.set_epoch(0)
    batch_a = next(iter(loader))
    aug_dataset.set_epoch(0)
    batch_b = next(iter(loader))
    if not torch.equal(batch_a[0], batch_b[0]):
        raise RuntimeError(
            "DeterministicAugDataset sanity check FAILED: "
            "same epoch produced different augmentations. "
            "Check that num_workers=0 or worker seeds are handled."
        )
    print(f"  [aug sanity check] PASSED — first {n_check} samples identical across two passes")


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',         type=str,   required=True)
    p.add_argument('--data_dir',        type=str,   default='/content/data')
    p.add_argument('--dataset',         type=str,   default='cifar10',
                   choices=['cifar10', 'cifar100'])
    p.add_argument('--model',           type=str,   default='resnet18')
    p.add_argument('--epochs',          type=int,   default=100)
    p.add_argument('--batch_size',      type=int,   default=256)
    p.add_argument('--lr',              type=float, default=0.1)
    p.add_argument('--seed',            type=int,   default=0)
    p.add_argument('--k',               type=int,   default=20)
    p.add_argument('--val_size',        type=int,   default=5000)
    p.add_argument('--patience',        type=int,   default=10)
    p.add_argument('--smoke_test',      action='store_true')
    p.add_argument('--wandb_project',   type=str,   default='cifar')
    p.add_argument('--run_name',        type=str,   default=None)
    p.add_argument('--no_animation',    action='store_true')
    p.add_argument('--log_img_every',   type=int,   default=10)

    # Reweighting
    p.add_argument('--strategy',        type=str,   default='both',
                   choices=['upweight_hard', 'downweight_easy', 'both'])
    p.add_argument('--up_factor',       type=float, default=1.1)
    p.add_argument('--down_factor',     type=float, default=0.9)
    p.add_argument('--soft_power',      type=float, default=2.0,
                   help='Power curve exponent (1=linear, 2=squared, higher=concentrate on hardest)')
    p.add_argument('--max_weight',      type=float, default=5.0)
    p.add_argument('--inclass_weights', action='store_true',
                   help='Restrict kNN to same class (recommended for CIFAR-100)')
    return p.parse_args()


# ── Weight computation ─────────────────────────────────────────────────────────

def compute_weights(val_losses, val_labels, val_embs,
                    train_losses, train_labels, train_embs,
                    n_train, args):
    weights  = np.ones(n_train, dtype=np.float32)
    n_classes = 100 if args.dataset == 'cifar100' else 10

    if args.inclass_weights:
        neighbor_indices = compute_within_class_neighbors_faiss(
            val_embs, val_labels, train_embs, train_labels,
            k=args.k, n_classes=n_classes)
    else:
        _, neighbor_indices = compute_neighbors_faiss(val_embs, train_embs, k=args.k)

    valid_losses = val_losses
    loss_min     = valid_losses.min()
    loss_range   = valid_losses.max() - loss_min + 1e-8

    n_upweighted = n_downweighted = 0

    for i in range(len(val_losses)):
        nbr_idx = neighbor_indices[i]
        nbr_idx = nbr_idx[nbr_idx >= 0]
        if len(nbr_idx) == 0:
            continue

        score = ((val_losses[i] - loss_min) / loss_range) ** args.soft_power

        if args.strategy in ('upweight_hard', 'both'):
            weights[nbr_idx] *= (1.0 + (args.up_factor   - 1.0) * score)
            n_upweighted     += 1
        if args.strategy in ('downweight_easy', 'both'):
            weights[nbr_idx] *= (1.0 - (1.0 - args.down_factor) * (1.0 - score))
            n_downweighted   += 1

    weights = np.clip(weights, 1.0 / args.max_weight, args.max_weight)
    weights = weights * (n_train / weights.sum())

    stats = {
        'weight_mean':    float(weights.mean()),
        'weight_std':     float(weights.std()),
        'weight_max':     float(weights.max()),
        'weight_min':     float(weights.min()),
        'n_upweighted':   int(n_upweighted),
        'n_downweighted': int(n_downweighted),
    }
    return weights, stats


# ── Augmented embedding collection ────────────────────────────────────────────

def collect_aug_embeddings(aug_dataset, train_indices, model, criterion,
                           device, batch_size, n_train):
    """
    Eval pass over aug_dataset (model frozen) collecting embeddings, losses,
    labels scattered to their original train positions via sample index.
    """
    loader = DataLoader(aug_dataset, batch_size=batch_size * 2,
                        shuffle=False, num_workers=0, pin_memory=True)
    emb_dim = None
    all_embs   = None
    all_losses = np.zeros(n_train, dtype=np.float32)
    all_labels = np.zeros(n_train, dtype=np.int64)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels, indices in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, emb    = model(inputs, return_embedding=True)
            losses         = criterion(logits, labels)

            if emb_dim is None:
                emb_dim  = emb.shape[1]
                all_embs = np.zeros((n_train, emb_dim), dtype=np.float32)

            idx = indices.numpy()
            all_embs[idx]   = emb.cpu().numpy()
            all_losses[idx] = losses.cpu().numpy()
            all_labels[idx] = labels.cpu().numpy()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total
    return all_losses, all_embs, all_labels, acc


# ── Per-class / superclass accuracy ───────────────────────────────────────────

def compute_per_class_acc(test_losses, test_labels, n_classes, classes):
    out = {}
    for cls in range(n_classes):
        mask = test_labels == cls
        if mask.sum() > 0:
            out[f'test_acc_{classes[cls]}'] = float((test_losses[mask] < np.log(2)).mean())
    return out


def compute_superclass_acc(test_losses, test_labels):
    out = {}
    for sc_name, fine_classes in CIFAR100_SUPERCLASSES.items():
        mask = np.isin(test_labels, fine_classes)
        if mask.sum() > 0:
            out[f'test_acc_sc_{sc_name}'] = float((test_losses[mask] < np.log(2)).mean())
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    n_classes = 100 if args.dataset == 'cifar100' else 10

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    run_name = args.run_name or (
        f"v2_{args.dataset}_{args.strategy}_pow{args.soft_power}_s{args.seed}"
    )
    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args), dir=args.out_dir)

    # ── Build datasets ─────────────────────────────────────────────────────
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

    DS = torchvision.datasets.CIFAR100 if args.dataset == 'cifar100' \
         else torchvision.datasets.CIFAR10

    full_train_raw  = DS(root=args.data_dir, train=True,  download=True)
    full_train_eval = DS(root=args.data_dir, train=True,  download=True,
                         transform=transform_eval)
    test_dataset    = DS(root=args.data_dir, train=False, download=True,
                         transform=transform_eval)

    # Fixed stratified split (seed=42, same as data.py)
    rng     = np.random.RandomState(42)
    targets = np.array(full_train_raw.targets)
    train_indices, val_indices = [], []
    val_per_class = args.val_size // n_classes
    for cls in range(n_classes):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        val_indices.extend(cls_idx[:val_per_class].tolist())
        train_indices.extend(cls_idx[val_per_class:].tolist())

    if args.smoke_test:
        train_indices = train_indices[:1000]
        val_indices   = val_indices[:200]

    n_train   = len(train_indices)
    test_indices = list(range(len(test_dataset))) if not args.smoke_test else list(range(200))
    classes   = full_train_raw.classes

    print(f"Split — Train: {n_train}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # DeterministicAugDataset over the train split
    aug_dataset = DeterministicAugDataset(
        Subset(full_train_raw, train_indices), transform_train)

    # Standard shuffled training loader (epoch 0 only, uniform weights)
    train_loader_e0 = DataLoader(
        aug_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True)

    # Clean eval loaders (val + test never use augmentation)
    ebs = args.batch_size * 2
    val_loader = DataLoader(
        Subset(full_train_eval, val_indices),
        batch_size=ebs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=ebs, shuffle=False, num_workers=0, pin_memory=True)

    # ── Augmentation sanity check ──────────────────────────────────────────
    print("Running augmentation determinism check...")
    verify_determinism(aug_dataset, n_check=min(64, n_train))

    # ── Model / optimizer ──────────────────────────────────────────────────
    model     = get_model(args.model, num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    sample_weights       = np.ones(n_train, dtype=np.float32)
    train_embs_history   = []
    train_losses_history = []
    train_labels_arr     = None
    displacements        = []
    umap_reducer         = None
    warmup_buffer        = []
    best_test_acc        = 0.0
    patience_counter     = 0

    for epoch in range(args.epochs):

        aug_dataset.set_epoch(epoch)

        # ── Step 1: collect augmented train embeddings + compute weights ───
        # (skipped on epoch 0 — no previous model to trust yet)
        weight_stats = {
            'weight_mean': 1.0, 'weight_std': 0.0,
            'weight_max':  1.0, 'weight_min': 1.0,
            'n_upweighted': 0,  'n_downweighted': 0,
        }

        if epoch > 0:
            t0 = time.time()
            train_losses, train_embs, train_labels, train_acc_aug = \
                collect_aug_embeddings(aug_dataset, train_indices, model,
                                       criterion, device,
                                       args.batch_size, n_train)
            print(f"  [aug embed {time.time()-t0:.1f}s] train_acc(aug)={train_acc_aug:.3f}")

            val_losses, val_embs, val_labels, val_acc = \
                evaluation_pass(val_loader, model, criterion, device, name='val  ')

            sample_weights, weight_stats = compute_weights(
                val_losses, val_labels, val_embs,
                train_losses, train_labels, train_embs,
                n_train=n_train, args=args)

            if train_labels_arr is None:
                train_labels_arr = train_labels
            prev_train_embs = train_embs_history[-1] if train_embs_history else None
            if prev_train_embs is not None:
                displacements.append(compute_displacement(prev_train_embs, train_embs))
            train_embs_history = [train_embs]
            train_losses_history.append(train_losses)
        else:
            prev_train_embs = None
            val_losses, val_embs, val_labels, val_acc = \
                evaluation_pass(val_loader, model, criterion, device, name='val  ')

        # ── Step 2: training pass — same augmentation, apply weights ───────
        model.train()
        weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(device)

        # Use a new DataLoader each epoch with shuffle; aug_dataset.epoch is fixed
        train_loader = DataLoader(
            aug_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=False)

        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1:03d}/{args.epochs} [Train]",
                         leave=False)
        for inputs, labels, indices in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits     = model(inputs)
            per_sample = criterion(logits, labels)
            loss       = (per_sample * weights_tensor[indices]).mean()
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()

        # ── Step 3: clean test eval ────────────────────────────────────────
        test_losses, test_embs, test_labels, test_acc = \
            evaluation_pass(test_loader, model, criterion, device, name='test ')

        # ── Step 4: metrics (use augmented embs when available) ───────────
        if epoch > 0:
            metrics, _, _, _ = compute_metrics_summary(
                val_losses, val_labels, val_embs,
                train_losses, train_labels, train_embs,
                prev_train_embs=prev_train_embs,
                k=args.k, n_classes=n_classes)
        else:
            metrics = {k: float('nan') for k in [
                'rho_global', 'rho_within', 'purity_mean', 'purity_median',
                'n_displaced', 'val_loss_mean', 'val_loss_std',
                'n_hard', 'n_easy', 'displacement_mean',
                'displacement_std', 'displacement_max',
            ]}

        # ── Step 5: UMAP images ────────────────────────────────────────────
        if not args.no_animation and epoch > 0:
            umap_fit_epoch = 3
            if epoch < umap_fit_epoch:
                warmup_buffer.append((train_embs.copy(), train_labels.copy(),
                                      train_losses.copy(), sample_weights.copy()))
            elif epoch == umap_fit_epoch:
                umap_reducer = fit_umap_reducer(train_embs)
                for past_ep, (e, lb, ls, w) in enumerate(warmup_buffer):
                    save_epoch_plots(umap_reducer, e, lb, ls, w,
                                     args.out_dir, past_ep + 1, n_classes=n_classes)
                warmup_buffer = []
            if umap_reducer is not None:
                plot_path = save_epoch_plots(
                    umap_reducer, train_embs, train_labels,
                    train_losses, sample_weights,
                    args.out_dir, epoch + 1, n_classes=n_classes)
                if args.log_img_every > 0 and (epoch + 1) % args.log_img_every == 0:
                    wandb.log({'embedding': wandb.Image(plot_path)}, step=epoch + 1)

        # ── Step 6: W&B log ────────────────────────────────────────────────
        per_class_acc  = compute_per_class_acc(test_losses, test_labels, n_classes, classes)
        superclass_acc = compute_superclass_acc(test_losses, test_labels) \
                         if args.dataset == 'cifar100' else {}

        wandb.log({
            'epoch':      epoch + 1,
            'val_loss':   float(val_losses.mean()),
            'test_loss':  float(test_losses.mean()),
            'val_acc':    val_acc,
            'test_acc':   test_acc,
            'lr':         scheduler.get_last_lr()[0],
            **metrics,
            **weight_stats,
            **per_class_acc,
            **superclass_acc,
        })

        rho_str = f"rho_w: {metrics['rho_within']:.3f}" if epoch > 0 else "rho_w: ---"
        print(f"Epoch {epoch+1:03d} | Val: {val_acc:.3f} | Test: {test_acc:.3f} | "
              f"{rho_str} | w_max: {weight_stats['weight_max']:.3f}")

        # ── Early stopping on test acc ─────────────────────────────────────
        if test_acc > best_test_acc:
            best_test_acc    = test_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (best: {best_test_acc:.3f})")
            break

    # ── End of training ────────────────────────────────────────────────────
    if not args.no_animation and displacements:
        make_displacement_plot(
            displacements, train_labels_arr, train_losses_history,
            out_path=os.path.join(args.out_dir, 'displacement.png'),
            n_classes=n_classes,
            classes=classes if n_classes <= 20 else None,
        )

    wandb.log({'best_test_acc': best_test_acc})
    wandb.finish()
    print(f"Done. Best test acc: {best_test_acc:.3f}")


if __name__ == '__main__':
    main()
