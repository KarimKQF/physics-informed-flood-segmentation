# -*- coding: utf-8 -*-
"""
Sequential launcher for the three full-dataset SegMAN-S seed0 baselines:
    1. CE          (full_seed0_ce)
    2. Dice        (full_seed0_dice)
    3. Dice+CE     (full_seed0_dice_ce)

Usage:
    python experiments_cvpr/segman/tools/launch_full_baselines.py

Each run saves:
  E:/flood_research/experiments/segman/runs/full_seed0_{ce,dice,dice_ce}/
    checkpoints/best_checkpoint.pt
    checkpoints/last_checkpoint.pt
    configs/full_seed0_{ce,dice,dice_ce}.yaml  (config copy)
    logs/{tag}_training.log
    metrics/training_epoch_metrics.csv
    metrics/{tag}_summary.json

Master log: E:/flood_research/logs/full_sen1floods11_baselines_master_stdout.log
"""

import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

VENV_PYTHON = Path(r"E:\flood_research\venvs\terramind-gpu\Scripts\python.exe")
TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "train_segman.py"
CFG_DIR      = Path(__file__).resolve().parents[3] / "configs" / "segman" / "full_baselines"

RUNS = [
    {"cond": "ce",      "cfg": CFG_DIR / "full_seed0_ce.yaml"},
    {"cond": "dice",    "cfg": CFG_DIR / "full_seed0_dice.yaml"},
    {"cond": "dice_ce", "cfg": CFG_DIR / "full_seed0_dice_ce.yaml"},
]

MASTER_LOG = Path("E:/flood_research/logs/full_sen1floods11_baselines_master_stdout.log")
MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)

STATUS_FILE = Path("E:/flood_research/experiments/segman/full_baselines_chain_status.json")
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str, fh=None) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def main() -> None:
    import json

    with MASTER_LOG.open("w", encoding="utf-8") as master:
        log(f"=== Full Sen1Floods11 Sequential Baseline Launcher ===", master)
        log(f"Runs: {[r['cond'] for r in RUNS]}", master)

        results = {}
        for run in RUNS:
            cond = run["cond"]
            cfg  = run["cfg"]
            log(f"\n{'='*50}", master)
            log(f"Starting: {cond}  config={cfg}", master)
            t0 = time.time()

            cmd = [str(VENV_PYTHON), str(TRAIN_SCRIPT), "--config", str(cfg)]
            log(f"CMD: {' '.join(cmd)}", master)

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            lines = []
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                master.write(line)
                master.flush()
                lines.append(line.rstrip())

            proc.wait()
            elapsed = time.time() - t0
            rc = proc.returncode

            result = {
                "condition": cond,
                "config": str(cfg),
                "returncode": rc,
                "elapsed_s": round(elapsed, 1),
                "status": "done" if rc == 0 else "failed",
            }
            # Extract best_epoch and best_val_miou from last INFO lines
            for line in reversed(lines):
                if "best_epoch=" in line and "best_val_miou=" in line:
                    import re
                    m = re.search(r"best_epoch=(\d+).*best_val_miou=([0-9.]+)", line)
                    if m:
                        result["best_epoch"]    = int(m.group(1))
                        result["best_val_miou"] = float(m.group(2))
                    break

            results[cond] = result
            STATUS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

            if rc == 0:
                log(f"DONE {cond}: elapsed={elapsed/3600:.2f}h best_miou={result.get('best_val_miou','?')}", master)
            else:
                log(f"FAILED {cond}: returncode={rc} — aborting chain.", master)
                break

        log(f"\n=== Chain complete ===", master)
        for cond, r in results.items():
            log(f"  {cond}: {r['status']}  mIoU={r.get('best_val_miou','?')}  epoch={r.get('best_epoch','?')}  {r['elapsed_s']/3600:.2f}h", master)
        STATUS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nMaster log: {MASTER_LOG}")
    print(f"Status: {STATUS_FILE}")


if __name__ == "__main__":
    main()
