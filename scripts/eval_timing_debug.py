"""
One-shot timing diagnostic for the full per-epoch eval block.
Uses a saved checkpoint so training doesn't need to run.

Usage:
    python scripts/eval_timing_debug.py \
        --checkpoint outputs/sweep2/rw_hard_only/best_model.pt \
        --config     outputs/sweep2/rw_hard_only/config.json
"""

import os
import warnings
os.environ.setdefault('PYTHONWARNINGS', 'ignore::FutureWarning')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)

import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data    import get_dataloaders
from utils.model   import get_model
from utils.metrics import (evaluation_pass, compute_neighbors_faiss,
                            compute_purity, compute_metrics_summary)


def tick(label, t0):
    elapsed = time.perf_counter() - t0
    print(f"  [{label}]  {elapsed:.2f}s")
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    (_, _, train_eval_loader, val_loader, test_loader, split_info) = get_dataloaders(
        data_dir        = cfg['data_dir'],
        val_size        = cfg['val_size'],
        batch_size      = cfg['batch_size'],
        seed            = cfg['seed'],
        smoke_test      = cfg['smoke_test'],
        eval_batch_size = cfg['batch_size'] * 2,
    )
    tick('data setup', t0)
    n_train = split_info['n_train']

    # ── Model ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model = get_model(cfg['model']).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    tick('model load', t0)

    criterion = nn.CrossEntropyLoss(reduction='none')

    print("\n── Evaluation passes ────────────────────────────────────────────")
    t_eval_total = time.perf_counter()

    t0 = time.perf_counter()
    train_losses, train_embs, train_labels, train_acc = \
        evaluation_pass(train_eval_loader, model, criterion, device, name='train')
    tick('train eval wall', t0)

    t0 = time.perf_counter()
    val_losses, val_embs, val_labels, val_acc = \
        evaluation_pass(val_loader, model, criterion, device, name='val  ')
    tick('val eval wall', t0)

    t0 = time.perf_counter()
    test_losses, test_embs, test_labels, test_acc = \
        evaluation_pass(test_loader, model, criterion, device, name='test ')
    tick('test eval wall', t0)

    tick('ALL eval passes', t_eval_total)

    print("\n── Post-eval: compute_weights ───────────────────────────────────")
    t0 = time.perf_counter()
    _, neighbor_indices = compute_neighbors_faiss(val_embs, train_embs, k=cfg['k'])
    tick('FAISS kNN (val→train)', t0)

    t0 = time.perf_counter()
    purity = compute_purity(val_labels, train_labels, neighbor_indices, k=cfg['k'])
    tick('purity', t0)

    t0 = time.perf_counter()
    weights = np.ones(n_train, dtype=np.float32)
    hard_threshold = np.percentile(val_losses, cfg['hard_percentile'])
    for i in range(len(val_losses)):
        if val_losses[i] >= hard_threshold:
            weights[neighbor_indices[i]] *= cfg['up_factor']
    weights = np.clip(weights, 1.0 / cfg['max_weight'], cfg['max_weight'])
    weights = weights * (n_train / weights.sum())
    tick('weight accumulation loop', t0)

    print("\n── Post-eval: compute_metrics_summary ───────────────────────────")
    t0 = time.perf_counter()
    metrics, _, _, _ = compute_metrics_summary(
        val_losses, val_labels, val_embs,
        train_losses, train_labels, train_embs,
        prev_train_embs=None,
        k=cfg['k'],
    )
    tick('compute_metrics_summary', t0)

    print(f"\nTrain acc: {train_acc:.3f}  Val acc: {val_acc:.3f}  Test acc: {test_acc:.3f}")


if __name__ == '__main__':
    main()
