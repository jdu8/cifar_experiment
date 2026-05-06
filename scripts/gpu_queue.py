"""
Parallel GPU job launcher for Colab.

Usage — put this at the top of your Colab run cell:

    from google.colab import userdata
    import os
    os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')

Then build your commands list and call GPUQueue(commands).run()
"""

import subprocess
import time
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False


def _gpu_mem_used_mb():
    if not _NVML:
        return None
    total = 0
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        total += info.used / 1024 ** 2
    return total


@dataclass
class Job:
    cmd: str
    index: int
    log_path: str
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    started_at: float = field(default_factory=time.time)
    stability_credited: bool = False

    def poll(self):
        return self.proc.poll() if self.proc else None

    def age(self):
        return time.time() - self.started_at

    def kill(self):
        if self.proc:
            self.proc.kill()


class GPUQueue:
    def __init__(self, commands, log_dir="/content/logs",
                 start_concurrency=1, stability_window=10.0,
                 poll_interval=2.0, crash_exit_codes=None):
        """
        commands          — list of shell command strings
        log_dir           — each job writes to log_dir/job_{n:03d}.log
        start_concurrency — how many jobs to run in parallel initially
        stability_window  — seconds a job must run before concurrency bumps up
        poll_interval     — seconds between status checks
        crash_exit_codes  — set of exit codes treated as crashes (None = any non-zero)
        """
        self.queue = deque(enumerate(commands))
        self.total = len(commands)
        self.log_dir = log_dir
        self.concurrency = start_concurrency
        self.max_safe_concurrency = None
        self.stability_window = stability_window
        self.poll_interval = poll_interval
        self.crash_exit_codes = crash_exit_codes
        self._running = []
        self._done = []

        os.makedirs(log_dir, exist_ok=True)

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        gpu = _gpu_mem_used_mb()
        gpu_str = f"  GPU:{gpu:.0f}MB" if gpu is not None else ""
        print(f"[{ts}]{gpu_str}  {msg}", flush=True)

    def _is_crash(self, code):
        if code == 0:
            return False
        if self.crash_exit_codes is None:
            return True
        return code in self.crash_exit_codes

    def _launch(self, n, cmd):
        log_path = os.path.join(self.log_dir, f"job_{n:03d}.log")
        self._log(f"▶  [job {n}]  running={len(self._running)+1}  log={log_path}")
        self._log(f"   cmd: {cmd}")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self._running.append(
            Job(cmd=cmd, index=n, log_path=log_path, proc=proc)
        )

    def _requeue(self, job):
        self.queue.appendleft((job.index, job.cmd))

    def tail(self, n=20):
        """Print the last n lines of every active log."""
        for job in self._running:
            if not os.path.exists(job.log_path):
                continue
            with open(job.log_path) as f:
                lines = f.readlines()
            print(f"\n── job {job.index} (age {job.age():.0f}s) ──")
            print("".join(lines[-n:]), end="")

    def run(self):
        self._log(f"GPUQueue  jobs={self.total}  concurrency={self.concurrency}")
        self._log(f"Logs in {self.log_dir}")

        while self.queue and len(self._running) < self.concurrency:
            n, cmd = self.queue.popleft()
            self._launch(n, cmd)

        all_ok = True
        while self.queue or self._running:
            time.sleep(self.poll_interval)
            still_running = []
            crashed = False

            for job in self._running:
                code = job.poll()

                if code is None:
                    still_running.append(job)
                    if (not job.stability_credited
                            and job.age() >= self.stability_window
                            and self.max_safe_concurrency is None
                            and not crashed):
                        job.stability_credited = True
                        self.concurrency += 1
                        self._log(f"✔  [job {job.index}] stable → concurrency {self.concurrency}")
                    continue

                elapsed = job.age()

                if self._is_crash(code):
                    tail = ""
                    try:
                        with open(job.log_path) as f:
                            lines = f.readlines()
                        tail = "".join(lines[-20:])
                    except Exception:
                        pass
                    self._log(
                        f"CRASH  [job {job.index}] exit={code} after={elapsed:.1f}s\n"
                        f"  log: {job.log_path}\n{tail.strip()}"
                    )
                    crashed = True

                    if self.concurrency > 1:
                        old = self.concurrency
                        self.concurrency -= 1
                        self.max_safe_concurrency = self.concurrency
                        self._log(f"concurrency locked {old} → {self.concurrency}")
                        for sibling in still_running:
                            sibling.kill()
                            self._requeue(sibling)
                        still_running = []
                    else:
                        self._log("concurrency=1 still crashing — skipping job")
                        all_ok = False
                    self._requeue(job)
                else:
                    self._log(f"DONE  [job {job.index}] exit={code} after={elapsed:.1f}s")
                    self._done.append(job)

            self._running = still_running
            while self.queue and len(self._running) < self.concurrency:
                n, cmd = self.queue.popleft()
                self._launch(n, cmd)

        self._log(f"Done  completed={len(self._done)}/{self.total}")
        return all_ok
