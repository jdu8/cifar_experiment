# scripts/train_reweight.py

import os
import warnings

# Must be set before DataLoader workers spawn so they inherit the filter.
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data      import get_dataloaders
from utils.model     import get_model
from utils.metrics   import (evaluation_pass, compute_metrics_summary,
                              compute_neighbors_faiss,
                              compute_within_class_neighbors_faiss,
                              compute_purity, compute_displacement)
from utils.animation import (fit_umap_reducer, save_epoch_plots,
                              make_displacement_plot)

# CIFAR-100 superclass → fine-class index mapping
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',          type=str,   required=True)
    p.add_argument('--data_dir',         type=str,   default='/content/data')
    p.add_argument('--dataset',          type=str,   default='cifar10',
                   choices=['cifar10', 'cifar100'])
    p.add_argument('--model',            type=str,   default='resnet18')
    p.add_argument('--epochs',           type=int,   default=100)
    p.add_argument('--batch_size',       type=int,   default=256)
    p.add_argument('--lr',               type=float, default=0.1)
    p.add_argument('--seed',             type=int,   default=0)
    p.add_argument('--k',                type=int,   default=20)
    p.add_argument('--val_size',         type=int,   default=5000)
    p.add_argument('--patience',         type=int,   default=10)
    p.add_argument('--smoke_test',       action='store_true')
    p.add_argument('--wandb_project',    type=str,   default='cifar')
    p.add_argument('--run_name',         type=str,   default=None)
    p.add_argument('--no_animation',     action='store_true')
    p.add_argument('--umap_fit_epoch',   type=int,   default=3)
    p.add_argument('--umap_max_points',  type=int,   default=20000)
    p.add_argument('--log_img_every',    type=int,   default=10,
                   help='Log UMAP image to W&B every N epochs (0 = never)')

    # Reweighting
    p.add_argument('--strategy',         type=str,   default='both',
                   choices=['upweight_hard', 'downweight_easy', 'both'])
    p.add_argument('--up_factor',        type=float, default=1.1)
    p.add_argument('--down_factor',      type=float, default=0.9)
    p.add_argument('--hard_percentile',  type=float, default=80.0)
    p.add_argument('--easy_percentile',  type=float, default=20.0)
    p.add_argument('--purity_threshold', type=float, default=0.15)
    p.add_argument('--purity_gate',      action='store_true')
    p.add_argument('--max_weight',       type=float, default=5.0)
    p.add_argument('--ema_alpha',        type=float, default=1.0,
                   help='EMA blend: 1.0=no smoothing, 0.3=heavy smoothing')
    p.add_argument('--soft_weights',     action='store_true',
                   help='Continuous loss-proportional scoring instead of binary threshold')
    p.add_argument('--soft_power',       type=float, default=1.0,
                   help='Power curve exponent for soft scoring (>1 concentrates on hardest tail)')
    p.add_argument('--inclass_weights',  action='store_true',
                   help='Use within-class neighbors for weight signal (recommended for CIFAR-100)')

    return p.parse_args()


def compute_weights(val_losses, val_labels, val_embs,
                    train_losses, train_labels, train_embs,
                    n_train, args):
    weights = np.ones(n_train, dtype=np.float32)

    if args.inclass_weights:
        neighbor_indices = compute_within_class_neighbors_faiss(
            val_embs, val_labels, train_embs, train_labels,
            k=args.k, n_classes=(100 if args.dataset == 'cifar100' else 10))
    else:
        _, neighbor_indices = compute_neighbors_faiss(val_embs, train_embs, k=args.k)

    if args.purity_gate and not args.inclass_weights:
        purity     = compute_purity(val_labels, train_labels, neighbor_indices, k=args.k)
        valid_mask = purity >= args.purity_threshold
    else:
        valid_mask = np.ones(len(val_losses), dtype=bool)

    n_displaced = (~valid_mask).sum()

    valid_losses   = val_losses[valid_mask]
    loss_min       = valid_losses.min()
    loss_max       = valid_losses.max()
    loss_range     = loss_max - loss_min + 1e-8
    hard_threshold = np.percentile(valid_losses, args.hard_percentile)
    easy_threshold = np.percentile(valid_losses, args.easy_percentile)

    n_upweighted = n_downweighted = 0

    for i in range(len(val_losses)):
        if not valid_mask[i]:
            continue

        nbr_idx = neighbor_indices[i]
        nbr_idx = nbr_idx[nbr_idx >= 0]   # filter -1 padding from inclass search
        if len(nbr_idx) == 0:
            continue

        loss_i = val_losses[i]

        if args.soft_weights:
            score = ((loss_i - loss_min) / loss_range) ** args.soft_power

            if args.strategy in ('upweight_hard', 'both'):
                weights[nbr_idx] *= (1.0 + (args.up_factor   - 1.0) * score)
                n_upweighted     += 1

            if args.strategy in ('downweight_easy', 'both'):
                weights[nbr_idx] *= (1.0 - (1.0 - args.down_factor) * (1.0 - score))
                n_downweighted   += 1
        else:
            if args.strategy in ('upweight_hard', 'both'):
                if loss_i >= hard_threshold:
                    weights[nbr_idx] *= args.up_factor
                    n_upweighted     += 1

            if args.strategy in ('downweight_easy', 'both'):
                if loss_i <= easy_threshold:
                    weights[nbr_idx] *= args.down_factor
                    n_downweighted   += 1

    weights = np.clip(weights, 1.0 / args.max_weight, args.max_weight)
    weights = weights * (n_train / weights.sum())

    stats = {
        'weight_mean':    float(weights.mean()),
        'weight_std':     float(weights.std()),
        'weight_max':     float(weights.max()),
        'weight_min':     float(weights.min()),
        'n_displaced':    int(n_displaced),
        'n_upweighted':   int(n_upweighted),
        'n_downweighted': int(n_downweighted),
    }
    return weights, stats


def compute_per_class_acc(test_losses, test_labels, n_classes, classes):
    """Per-class accuracy using loss < log(2) as proxy (p_correct > 0.5)."""
    out = {}
    for cls in range(n_classes):
        mask = test_labels == cls
        if mask.sum() > 0:
            out[f'test_acc_{classes[cls]}'] = float((test_losses[mask] < np.log(2)).mean())
    return out


def compute_superclass_acc(test_losses, test_labels):
    """Per-superclass accuracy for CIFAR-100."""
    out = {}
    for sc_name, fine_classes in CIFAR100_SUPERCLASSES.items():
        mask = np.isin(test_labels, fine_classes)
        if mask.sum() > 0:
            out[f'test_acc_sc_{sc_name}'] = float((test_losses[mask] < np.log(2)).mean())
    return out


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_classes = 100 if args.dataset == 'cifar100' else 10

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    run_name = args.run_name or (
        f"reweight_{args.dataset}_{args.strategy}"
        f"_up{args.up_factor}_s{args.seed}"
    )

    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args), dir=args.out_dir)

    (train_loader, train_loader_indexed,
     train_eval_loader, val_loader,
     test_loader, split_info) = get_dataloaders(
        data_dir=args.data_dir,
        val_size=args.val_size,
        batch_size=args.batch_size,
        seed=args.seed,
        smoke_test=args.smoke_test,
        dataset=args.dataset,
    )

    n_train = split_info['n_train']
    classes = split_info['classes']   # list of class name strings

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
    warmup_buffer        = []   # (embs, labels, losses, weights) before UMAP fit
    best_test_acc        = 0.0
    patience_counter     = 0

    for epoch in range(args.epochs):

        # ── Training pass ─────────────────────────────────────────────────
        model.train()
        weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(device)

        train_bar = tqdm(
            train_loader_indexed,
            total=len(train_loader_indexed),
            desc=f"Epoch {epoch+1:03d}/{args.epochs} [Train]",
            leave=False
        )
        for inputs, labels, indices in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits      = model(inputs)
            per_sample  = criterion(logits, labels)
            loss        = (per_sample * weights_tensor[indices]).mean()
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()

        # ── Evaluation passes ─────────────────────────────────────────────
        t_eval = time.time()
        train_losses, train_embs, train_labels, train_acc = \
            evaluation_pass(train_eval_loader, model, criterion, device, name='train')
        val_losses,   val_embs,   val_labels,   val_acc   = \
            evaluation_pass(val_loader,        model, criterion, device, name='val  ')
        test_losses,  test_embs,  test_labels,  test_acc  = \
            evaluation_pass(test_loader,       model, criterion, device, name='test ')
        print(f"  [eval {time.time()-t_eval:.1f}s]", end="  ")

        if train_labels_arr is None:
            train_labels_arr = train_labels

        prev_train_embs = train_embs_history[-1] if train_embs_history else None
        if prev_train_embs is not None:
            displacements.append(compute_displacement(prev_train_embs, train_embs))

        train_embs_history = [train_embs]
        train_losses_history.append(train_losses)

        # ── Compute new weights ───────────────────────────────────────────
        fresh_weights, weight_stats = compute_weights(
            val_losses, val_labels, val_embs,
            train_losses, train_labels, train_embs,
            n_train=n_train, args=args
        )
        if args.ema_alpha < 1.0:
            sample_weights  = args.ema_alpha * fresh_weights + (1.0 - args.ema_alpha) * sample_weights
            sample_weights  = sample_weights * (n_train / sample_weights.sum())
        else:
            sample_weights = fresh_weights

        # ── UMAP projections (3-panel image per epoch) ────────────────────
        if not args.no_animation:
            if epoch < args.umap_fit_epoch:
                warmup_buffer.append((train_embs.copy(), train_labels.copy(),
                                      train_losses.copy(), sample_weights.copy()))
            elif epoch == args.umap_fit_epoch:
                umap_reducer = fit_umap_reducer(
                    train_embs, max_points=args.umap_max_points)
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

        # ── Metrics ───────────────────────────────────────────────────────
        metrics, _, _, _ = compute_metrics_summary(
            val_losses, val_labels, val_embs,
            train_losses, train_labels, train_embs,
            prev_train_embs=prev_train_embs,
            k=args.k, n_classes=n_classes
        )

        per_class_acc   = compute_per_class_acc(test_losses, test_labels, n_classes, classes)
        superclass_acc  = compute_superclass_acc(test_losses, test_labels) \
                          if args.dataset == 'cifar100' else {}

        # ── W&B log ───────────────────────────────────────────────────────
        wandb.log({
            'epoch':      epoch + 1,
            'train_loss': float(train_losses.mean()),
            'val_loss':   float(val_losses.mean()),
            'test_loss':  float(test_losses.mean()),
            'train_acc':  train_acc,
            'val_acc':    val_acc,
            'test_acc':   test_acc,
            'lr':         scheduler.get_last_lr()[0],
            **metrics,
            **weight_stats,
            **per_class_acc,
            **superclass_acc,
        })

        print(f"Epoch {epoch+1:03d} | "
              f"Train: {train_acc:.3f} | "
              f"Val: {val_acc:.3f} | "
              f"Test: {test_acc:.3f} | "
              f"rho_w: {metrics['rho_within']:.3f} | "
              f"w_mean: {weight_stats['weight_mean']:.3f}")

        if test_acc > best_test_acc:
            best_test_acc    = test_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, 'best_model.pt'))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(best test acc: {best_test_acc:.3f})")
            break

    # ── End of training ───────────────────────────────────────────────────
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
