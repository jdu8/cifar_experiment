# CIFAR Reweighting Experiment — Change Backlog

## What this project does

Investigates whether using validation set difficulty signals to reweight training samples improves generalisation on CIFAR-10/CIFAR-100. The core idea: after each epoch, compute embeddings for train and val, find k nearest train neighbours for each val point (FAISS), and upweight train points near hard val examples / downweight those near easy ones.

The main training file is `scripts/train_reweight.py`. A cleaner version with aligned augmentations is `scripts/train_v2.py` (see below). Utilities are in `scripts/utils/`.

---

## File map

```
scripts/
  train_reweight.py       # current working training file
  train_v2.py             # new file: deterministic augmentation alignment (partial, see below)
  train.py                # DO NOT USE — broken imports, ignore this file
  test_deterministic_aug.py  # verification script for DeterministicAugDataset (passes)
  utils/
    data.py               # dataloaders, stratified train/val split from 50K
    model.py              # ResNetCIFAR wrapper + get_model()
    metrics.py            # evaluation_pass, FAISS kNN, Spearman rho, compute_metrics_summary
    animation.py          # UMAP fitting, 3-panel PNG per epoch, displacement plot
```

The 10K CIFAR test set is used as the true held-out test throughout. The 50K training set is split 4:1 into train (40K) and a reweighting-val (10K). The reweighting-val is what drives the weight signals — it is NOT the 10K test set.

---

## Changes to make (prioritised)

### 0. Verbose flag system + remove always-on timing prints — SIMPLE (metrics.py + train_reweight.py + train_v2.py)

**Current problem:** `evaluation_pass` in `metrics.py` (lines ~129-131) always prints a per-section timing breakdown for every eval pass:
```
train: load=16.60s  transfer=0.08s  forward=11.99s  gather=0.14s  total=28.81s  (40000 samples)
val  : load=4.65s  transfer=0.02s  forward=2.17s  gather=0.02s  total=6.85s  (10000 samples)
test : load=4.73s  transfer=0.01s  forward=2.18s  gather=0.02s  total=6.94s  (10000 samples)
```
This is noisy for normal runs. Remove the always-on print. The total eval time `[eval 42.8s]` printed in the training loop should stay.

**Add a `--verbose` argument** that accepts one or more of these modes (use `nargs='+'` or separate bool flags):

| Flag | What it prints |
|---|---|
| `--verbose_timing` | Per-section breakdown per eval pass (load / transfer / forward / gather). Also time taken per training phase (weight computation, FAISS search, UMAP if enabled). Currently these are always printed — move them behind this flag. |
| `--verbose_metrics` | Also print loss values alongside accuracy in the epoch line: `train_loss=1.23 val_loss=2.45 test_loss=2.44` |
| `--verbose_resources` | At end of each epoch: GPU memory used/total (via `pynvml` or `torch.cuda.memory_allocated()`), CPU RAM (via `psutil.virtual_memory()`). Poll once per epoch — overhead is ~5ms, acceptable. Do NOT poll per batch, that would hurt throughput. |

**Default output (no flags):**
```
[eval 42.8s]  Epoch 053 | Val: 0.647 | Test: 0.647 | rho_w: 0.825 | w_mean: 1.000
```

**With `--verbose_timing`:**
```
    train: load=16.60s  transfer=0.08s  forward=11.99s  gather=0.14s  total=28.81s  (40000 samples)
    val  : load=4.65s   transfer=0.02s  forward=2.17s   gather=0.02s  total=6.85s   (10000 samples)
    test : load=4.73s   transfer=0.01s  forward=2.18s   gather=0.02s  total=6.94s   (10000 samples)
    weight_compute=2.1s  faiss=0.4s
[eval 42.8s]  Epoch 053 | Val: 0.647 | Test: 0.647 | rho_w: 0.825 | w_mean: 1.000
```

**With `--verbose_metrics`:**
```
[eval 42.8s]  Epoch 053 | Val: 0.647 | Test: 0.647 | rho_w: 0.825 | w_mean: 1.000 | train_loss=0.23 val_loss=1.45 test_loss=1.44
```

**With `--verbose_resources`:**
```
[eval 42.8s]  Epoch 053 | Val: 0.647 | Test: 0.647 | rho_w: 0.825 | w_mean: 1.000
    resources: GPU 4.1/15.8GB  RAM 6.2/12.7GB
```

**Implementation notes:**
- Pass a `verbose_timing: bool` param into `evaluation_pass()` in `metrics.py` — the function already collects all timings, just conditionally prints them
- `torch.cuda.memory_allocated()` and `torch.cuda.get_device_properties(0).total_memory` need no extra deps
- `psutil` is already likely installed on Colab; add to requirements.txt if missing
- Note: watching resource usage via `pynvml` or `psutil` once per epoch has negligible performance impact (~5ms). Per-batch polling would matter — don't do that.

### 1. Remove early stopping — SIMPLE (train_reweight.py + train_v2.py)

Early stopping is currently on by default (`--patience 10`). Remove it entirely — runs should go to `--epochs` always. Delete the `patience_counter` logic and `--patience` argument from both files. If the user wants to stop early they can just kill the process.

**In `train_reweight.py`:** remove lines ~86-88 (patience arg), ~369-374 (patience counter increment/break).
**In `train_v2.py`:** same pattern, lines ~113-115 (patience arg) and ~214-219 (patience counter).

---

### 2. Make train/val split configurable — SIMPLE (data.py + train_reweight.py)

Currently `--val_size` controls how many of the 50K go to the reweighting-val set. The split is stratified (equal per class). Setting `--val_size 0` should mean all 50K go to training (pure baseline mode, no reweighting signal).

**In `scripts/utils/data.py`** (`get_dataloaders`, line 56 onwards): the split loop already uses `val_per_class = val_size // n_classes`. If `val_size=0`, `val_per_class=0` so all samples go to `train_indices` — this is correct. But `val_loader` will be empty, which will crash `evaluation_pass`. Add a guard: if `val_size == 0`, return `val_loader = None` and document it.

**In `train_reweight.py`**: add a guard around the weight computation block — if `val_loader is None`, skip `compute_weights` and keep `sample_weights = np.ones(n_train)`. This way `--val_size 0` is a clean no-reweighting baseline that trains on all 50K.

This means there is no need for a separate `train.py` for baselines. Just use `train_reweight.py --val_size 0 --up_factor 1.0 --down_factor 1.0`.

---

### 3. Add small CNN architectures — MEDIUM (model.py)

Add three CNN architectures smaller than ResNet18 (11M params). Wire them into `get_model()` so `--model tiny_cnn / small_cnn / medium_cnn` works.

These are for a POC phase on CIFAR-10 to verify the reweighting signal exists at all on a fast training setup (ResNet18 trains too slowly for rapid iteration). The embedding dimension intentionally differs across sizes — this is fine, FAISS and kNN work on any dimension.

**Architecture specs:**

```python
# Tiny (~150K params, embedding_dim=128)
# CIFAR-10 baseline target: ~75-80%
Conv2d(3,   32, 3, padding=1) + BN + ReLU + MaxPool(2)   # 16x16
Conv2d(32,  64, 3, padding=1) + BN + ReLU + MaxPool(2)   # 8x8
Conv2d(64, 128, 3, padding=1) + BN + ReLU + MaxPool(2)   # 4x4
AdaptiveAvgPool2d(1)  → flatten → Linear(128, n_classes)
embedding_dim = 128

# Small (~800K params, embedding_dim=256)
# CIFAR-10 baseline target: ~85-88%
Conv2d(3,   64, 3, padding=1) + BN + ReLU + MaxPool(2)   # 16x16
Conv2d(64, 128, 3, padding=1) + BN + ReLU + MaxPool(2)   # 8x8
Conv2d(128, 256, 3, padding=1) + BN + ReLU
Conv2d(256, 256, 3, padding=1) + BN + ReLU + MaxPool(2)  # 4x4
AdaptiveAvgPool2d(1)  → flatten → Linear(256, n_classes)
embedding_dim = 256

# Medium (~3M params, embedding_dim=512)
# CIFAR-10 baseline target: ~90-92%
Conv2d(3,   64, 3, padding=1) + BN + ReLU + MaxPool(2)   # 16x16
Conv2d(64, 128, 3, padding=1) + BN + ReLU + MaxPool(2)   # 8x8
Conv2d(128, 256, 3, padding=1) + BN + ReLU
Conv2d(256, 256, 3, padding=1) + BN + ReLU + MaxPool(2)  # 4x4
Conv2d(256, 512, 3, padding=1) + BN + ReLU
Conv2d(512, 512, 3, padding=1) + BN + ReLU + MaxPool(2)  # 2x2
AdaptiveAvgPool2d(1)  → flatten → Linear(512, n_classes)
embedding_dim = 512
```

The `forward(x, return_embedding=False)` interface must match `ResNetCIFAR` exactly — `metrics.py` calls `model(inputs, return_embedding=True)` and expects `(logits, emb)` back.

---

### 4. Distance-weighted kNN (Gaussian/cosine) — MEDIUM (metrics.py + train_reweight.py)

**Context:** currently we take k=20 nearest train neighbours per val point and upweight each uniformly. The k-th nearest neighbour (possibly far away) gets the same weight as the 1st. This is too blunt.

**The fix:** weight each neighbour's contribution by its cosine similarity to the val point. FAISS already returns similarity scores (inner product on normalised vectors = cosine similarity), so this is nearly free.

**In `compute_weights` in `train_reweight.py`** (the inner loop, lines ~130-160):

Instead of:
```python
weights[nbr_idx] *= (1.0 + (args.up_factor - 1.0) * score)
```

Use:
```python
# distances shape: (k,), values in [0, 1] since normalised
# already returned by compute_neighbors_faiss but currently discarded
sim_weights = distances[i]  # cosine similarities, already in [0,1]
sim_weights = sim_weights / (sim_weights.sum() + 1e-8)  # normalise
for j, tidx in enumerate(nbr_idx):
    weights[tidx] *= (1.0 + (args.up_factor - 1.0) * score * sim_weights[j] * k)
    # multiply by k to preserve the same overall magnitude as before
```

Or more simply: accumulate a weighted score per train point, then apply once after the loop (avoids repeated multiplication).

To do this properly, `compute_neighbors_faiss` needs to return distances alongside indices — it already does (`return distances, indices`) but `compute_weights` currently calls it as `_, neighbor_indices`. Pass distances through.

Add `--distance_weighted` flag (bool) to opt in. Default False to preserve current behaviour until tested.

---

### 5. train_v2.py — complete and verify

`train_v2.py` exists and has the core structure but needs:
- Changes 1 (no early stopping) applied
- Changes 2 (val_size=0 guard) applied
- Smoke test run to confirm it trains without errors

The key idea in train_v2.py: `DeterministicAugDataset` ensures that within an epoch, the embedding eval pass and the training pass use identical augmented images (seeded by `epoch * n_train + sample_idx`). This means weights are computed in the same augmented embedding space that training operates in.

`test_deterministic_aug.py` already verifies this works (all tests pass with num_workers=0).

---

## Experiment plan (after changes above)

### Phase 1 — CNN baseline hyperparameter sweep (CIFAR-10)

Goal: find best lr + batch_size per model size. No reweighting (val_size=0).

```bash
# Example for one config — repeat across lr in {0.01, 0.05, 0.1} and batch_size in {128, 256}
python scripts/train_reweight.py \
  --dataset cifar10 --model tiny_cnn \
  --val_size 0 --up_factor 1.0 --down_factor 1.0 \
  --epochs 50 --lr 0.05 --batch_size 128 \
  --no_animation \
  --out_dir outputs/hparam/tiny_lr005_bs128 \
  --run_name tiny_lr005_bs128 --wandb_project cifar_poc
```

3 models × 3 lr × 2 batch_size = 18 runs. Pick best lr+batch_size per model by final test_acc.

### Phase 2 — Reweighting vs baseline (best hyperparams, 3 seeds)

```bash
# Baseline: val_size=0, all 50K for training
# Reweight soft linear: val_size=10000, soft_power=1.0
# Reweight soft squared: val_size=10000, soft_power=2.0
```

Run each × 3 seeds. 3 strategies × 3 seeds × 3 models = 27 runs.

### Phase 3 — CIFAR-100 + ResNet18

Once the signal is confirmed on small models, replicate with ResNet18 + CIFAR-100. Same structure, val_size=10000, inclass_weights=True.

---

## Key decisions / context not in code

- **Why val_size=0 for baseline:** avoids having a separate train.py. train_reweight.py with val_size=0 skips all weight computation and trains on full 50K identically to a standard baseline.
- **Why no early stopping:** experiments run to fixed epoch count for fair comparison. Monitor test_acc curve in W&B to check for overfitting manually.
- **train.py is broken:** imports functions removed from animation.py. Do not try to fix or use it.
- **FAISS GPU:** install `faiss-gpu` on Colab (`pip install faiss-gpu-cu12`). Code already has GPU→CPU fallback in `metrics.py:_build_faiss_index`.
- **Animation is expensive:** always use `--no_animation` for sweeps. Enable only for a single interesting final run.
- **train_v2.py vs train_reweight.py:** train_v2.py is the more principled version (aligned augmentations) but train_reweight.py is tested and working. Run Phase 1+2 on train_reweight.py first to establish baselines, then re-run key configs on train_v2.py to measure whether aligned augs make a difference.
