"""
Colab sweep launcher.

Before running:
  1. Add WANDB_API_KEY to Colab Secrets (key icon, left sidebar)
  2. Mount Drive:  from google.colab import drive; drive.mount('/content/drive')
  3. Run this cell
"""

import sys

# WANDB_API_KEY must be set in the notebook before running this script:
#   from google.colab import userdata
#   import os; os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')

sys.path.insert(0, '/content/cifar_experiment/scripts')
from gpu_queue import GPUQueue  # noqa: E402

# ── paths ──────────────────────────────────────────────────────────────────
BASE     = "/content/cifar_experiment/scripts"
DATA     = "/content/data"
OUT_BASE = "/content/drive/MyDrive/cifar_experiment/outputs/sweep1"
LOG_DIR  = "/content/logs/sweep1"

def baseline(run_name, model='resnet18', epochs=100, seed=0, extra=''):
    return (
        f"python {BASE}/train.py "
        f"--out_dir {OUT_BASE}/{run_name} "
        f"--data_dir {DATA} "
        f"--model {model} "
        f"--epochs {epochs} --seed {seed} "
        f"--run_name {run_name} "
        f"--wandb_project cifar "
        f"{extra}"
    )

def reweight(run_name, strategy, up_factor=1.1, down_factor=0.9,
             purity_gate=False, epochs=100, seed=0, extra=''):
    gate_flag = '--purity_gate' if purity_gate else ''
    return (
        f"python {BASE}/train_reweight.py "
        f"--out_dir {OUT_BASE}/{run_name} "
        f"--data_dir {DATA} "
        f"--model resnet18 "
        f"--epochs {epochs} --seed {seed} "
        f"--strategy {strategy} "
        f"--up_factor {up_factor} --down_factor {down_factor} "
        f"{gate_flag} "
        f"--run_name {run_name} "
        f"--wandb_project cifar "
        f"{extra}"
    )

commands = [
    # ── Convergence + seed variance ────────────────────────────────────────
    baseline('rn18_100ep_s0',  seed=0),
    baseline('rn18_100ep_s1',  seed=1),
    baseline('rn18_100ep_s2',  seed=2),
    baseline('rn18_200ep_s0',  seed=0, epochs=200),

    # ── Model size ─────────────────────────────────────────────────────────
    baseline('rn34_100ep_s0',  model='resnet34'),
    baseline('rn50_100ep_s0',  model='resnet50'),
    baseline('rn101_100ep_s0', model='resnet101'),

    # ── Reweighting strategy ───────────────────────────────────────────────
    reweight('rw_hard_only',         strategy='upweight_hard',   purity_gate=False),
    reweight('rw_easy_only',         strategy='downweight_easy', purity_gate=False),
    reweight('rw_both',              strategy='both',            purity_gate=False),
    reweight('rw_both_gated',        strategy='both',            purity_gate=True),
    reweight('rw_both_gated_strong', strategy='both',            purity_gate=True,
             up_factor=1.2, down_factor=0.8),
]

GPUQueue(commands, log_dir=LOG_DIR).run()
