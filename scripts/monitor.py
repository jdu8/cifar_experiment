"""
Print the latest epoch line from each log file every 30 seconds.
Usage:  !python /content/cifar_experiment/scripts/monitor.py
        !python /content/cifar_experiment/scripts/monitor.py --log_dir /content/logs/sweep1
"""

import glob
import os
import time
import argparse
from datetime import datetime

LOG_DIR = "/content/logs/sweep1"


def last_epoch_line(path):
    """Return the last 'Epoch ...' line from a log file, or a status string."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return "  (unreadable)"

    for line in reversed(lines):
        if line.startswith("Epoch"):
            return "  " + line.rstrip()

    # Job may have just started or crashed before printing any epoch
    if lines:
        return "  (no epoch yet) last: " + lines[-1].rstrip()
    return "  (empty)"


def is_done(path):
    try:
        with open(path) as f:
            text = f.read()
        return "Done. Best test acc" in text or "Early stopping" in text
    except OSError:
        return False


def report(log_dir):
    logs = sorted(glob.glob(os.path.join(log_dir, "job_*.log")))
    if not logs:
        print("  No log files found in", log_dir)
        return

    done = sum(is_done(p) for p in logs)
    print(f"  {done}/{len(logs)} jobs done\n")

    for path in logs:
        name = os.path.basename(path)
        status = "DONE" if is_done(path) else "runs"
        print(f"  [{status}] {name}")
        print(last_epoch_line(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir', default=LOG_DIR)
    p.add_argument('--interval', type=int, default=30)
    args = p.parse_args()

    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'─'*50}  {ts}")
        report(args.log_dir)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
