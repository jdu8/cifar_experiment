"""
Colab run script — edit the commands list then run this cell.

Before running, add WANDB_API_KEY to Colab Secrets
(key icon in left sidebar) and enable notebook access.
"""

import os
import sys
from google.colab import userdata

os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')

sys.path.insert(0, '/content/cifar_experiment/scripts')
from gpu_queue import GPUQueue  # noqa: E402

# ── paths ──────────────────────────────────────────────────────────────────
BASE     = "/content/cifar_experiment/scripts"
DATA     = "/content/data"
OUT_BASE = "/content/drive/MyDrive/cifar_experiment/outputs/run1"
LOG_DIR  = "/content/logs"

# ── commands ───────────────────────────────────────────────────────────────
commands = [
    (f"python {BASE}/train.py "
     f"--out_dir {OUT_BASE}/baseline "
     f"--data_dir {DATA} "
     f"--epochs 100 --seed 0 "
     f"--run_name baseline "
     f"--wandb_project cifar"),

    (f"python {BASE}/train_reweight.py "
     f"--out_dir {OUT_BASE}/rw_both_gated "
     f"--data_dir {DATA} "
     f"--epochs 100 --seed 0 "
     f"--strategy both "
     f"--up_factor 1.1 --down_factor 0.9 "
     f"--purity_gate "
     f"--run_name rw_both_gated "
     f"--wandb_project cifar"),
]

GPUQueue(commands, log_dir=LOG_DIR).run()
