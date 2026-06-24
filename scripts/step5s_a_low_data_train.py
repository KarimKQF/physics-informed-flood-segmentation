"""
Config-driven low-data launcher for the validated STEP 5S-A bs2/accum4 runner.

This wrapper keeps the STEP 5S-A training implementation unchanged, but redirects
its run directory and split globals from a low-data YAML config so pilot runs do
not overwrite the full-data baseline artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import step5s_a_bs2_accum4_train as runner  # noqa: E402


DEFAULT_BOLIVIA_SPLIT = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step5e_tiny_unetdecoder_baseline/manifests/flood_bolivia_step5e_filtered.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle)


def configure_runner(config_path: Path, config: dict[str, Any]) -> None:
    run_dir = Path(config["run_dir"])
    data_args = config["data"]["init_args"]
    eval_splits = config.get("evaluation_splits", {})
    run_tag = config.get("run_tag", run_dir.name)

    runner.RUN_DIR = run_dir
    runner.CONFIG_PATH = config_path
    runner.SPLIT_FILES = {
        "train": Path(data_args["train_split"]),
        "valid": Path(data_args["val_split"]),
        "test": Path(data_args["test_split"]),
        "bolivia": Path(eval_splits.get("bolivia_split", DEFAULT_BOLIVIA_SPLIT)),
    }
    runner.EPOCH_CSV = run_dir / "metrics" / "training_epoch_metrics.csv"
    runner.SUMMARY_JSON = run_dir / "metrics" / f"{run_tag}_summary.json"
    runner.TRAINING_STATE = run_dir / "metrics" / "training_state.json"
    runner.BEST_CKPT = run_dir / "checkpoints" / "best_checkpoint.pt"
    runner.LAST_CKPT = run_dir / "checkpoints" / "last_checkpoint.pt"


def refuse_existing_artifacts(run_dir: Path, run_tag: str) -> None:
    artifacts = [
        run_dir / "checkpoints" / "best_checkpoint.pt",
        run_dir / "checkpoints" / "last_checkpoint.pt",
        run_dir / "metrics" / "training_epoch_metrics.csv",
        run_dir / "metrics" / "training_state.json",
        run_dir / "metrics" / f"{run_tag}_summary.json",
    ]
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to overwrite existing low-data STEP 5S-A artifacts: "
            + ", ".join(existing)
        )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_dir = Path(config["run_dir"])
    run_tag = config.get("run_tag", run_dir.name)

    refuse_existing_artifacts(run_dir, run_tag)
    configure_runner(args.config, config)

    forwarded = [str(Path(runner.__file__).name), "--config", str(args.config)]
    if args.resume is not None:
        forwarded.extend(["--resume", str(args.resume)])
    if args.log_file is not None:
        forwarded.extend(["--log-file", str(args.log_file)])
    sys.argv = forwarded
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
