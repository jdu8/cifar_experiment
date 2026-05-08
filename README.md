# CIFAR Reweighting Experiment

Investigates whether using validation set difficulty signals to reweight training samples improves generalisation. For each val point, we find its nearest neighbours in the training embedding space and upweight/downweight those neighbours based on how hard or easy the val point is. Tested on CIFAR-10 and CIFAR-100 with ResNet18.

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## How it works

Each epoch after a forward pass:
1. Compute embeddings for all train, val, and test samples
2. For each val point find its k nearest train neighbours (global or within-class)
3. Hard val points (high loss) → upweight their train neighbours
4. Easy val points (low loss) → downweight their train neighbours
5. Train next epoch with these per-sample weights

Weight scoring is either **binary** (hard/easy threshold) or **soft** (continuous loss-proportional score, optionally with a power curve to concentrate signal on the hardest tail).

## Project structure

```
scripts/
  train.py               # standard baseline training
  train_reweight.py      # reweighting training loop
  eval_timing_debug.py   # per-section timing diagnostic using a checkpoint
  utils/
    data.py              # dataloaders, train/val split
    model.py             # ResNet CIFAR variant
    metrics.py           # evaluation pass, FAISS kNN, Spearman rho
    animation.py         # UMAP fitting, per-epoch 3-panel PNG
```

## Experiments

### Baseline

```bash
python scripts/train_reweight.py \
    --out_dir outputs/cifar100/baseline_s0 \
    --data_dir data --dataset cifar100 \
    --model resnet18 --epochs 100 --seed 0 \
    --val_size 10000 --patience 200 \
    --strategy both --up_factor 1.0 --down_factor 1.0 \
    --soft_weights --soft_power 1.0 \
    --inclass_weights \
    --run_name baseline_c100_s0 --wandb_project cifar100
```

### Reweight — both up+down, squared scaling (recommended)

```bash
python scripts/train_reweight.py \
    --out_dir outputs/cifar100/rw_both_sq_s0 \
    --data_dir data --dataset cifar100 \
    --model resnet18 --epochs 100 --seed 0 \
    --val_size 10000 --patience 200 \
    --strategy both --up_factor 1.1 --down_factor 0.9 \
    --soft_weights --soft_power 2.0 \
    --inclass_weights \
    --run_name rw_both_sq_s0 --wandb_project cifar100
```

### Reweight — both up+down, linear scaling

```bash
python scripts/train_reweight.py \
    --out_dir outputs/cifar100/rw_both_lin_s0 \
    --data_dir data --dataset cifar100 \
    --model resnet18 --epochs 100 --seed 0 \
    --val_size 10000 --patience 200 \
    --strategy both --up_factor 1.1 --down_factor 0.9 \
    --soft_weights --soft_power 1.0 \
    --inclass_weights \
    --run_name rw_both_lin_s0 --wandb_project cifar100
```

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10` or `cifar100` |
| `--val_size` | `5000` | Validation set size (use 10000 for CIFAR-100) |
| `--strategy` | `both` | `upweight_hard`, `downweight_easy`, or `both` |
| `--up_factor` | `1.1` | Neighbour upweight multiplier |
| `--down_factor` | `0.9` | Neighbour downweight multiplier |
| `--soft_weights` | off | Continuous loss-proportional scoring instead of binary threshold |
| `--soft_power` | `1.0` | Power curve exponent — higher concentrates signal on hardest tail |
| `--inclass_weights` | off | Restrict neighbourhood search to same class (recommended for CIFAR-100) |
| `--purity_gate` | off | Skip val points whose neighbours are mostly wrong class |
| `--ema_alpha` | `1.0` | EMA blend across epochs — lower = more smoothing |
| `--k` | `20` | Neighbourhood size |
| `--patience` | `10` | Early stopping patience |
| `--no_animation` | off | Skip UMAP projection images |
| `--log_img_every` | `10` | Log UMAP image to W&B every N epochs |
