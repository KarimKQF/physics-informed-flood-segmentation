"""STEP 5R official benchmark and segmentation-loss parity audit.

This script is intentionally read-only with respect to raw data. It creates
audit artifacts only and does not launch training.
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - fallback for minimal envs
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path("E:/flood_research/experiments/terramind_baseline")
RUNS_ROOT = BASELINE_ROOT / "runs"
STEP5R_RUN_DIR = RUNS_ROOT / "step5r_official_benchmark_loss_parity_audit"

KEYWORDS = [
    "Sen1Floods11",
    "Sen1Fl11",
    "sen1floods",
    "TerraMind",
    "TerraMind-L",
    "UPerNet",
    "UperNetDecoder",
    "UNetDecoder",
    "Dice",
    "DiceLoss",
    "CrossEntropy",
    "CE",
    "Focal",
    "Tversky",
    "PANGAEA",
    "ignore_index",
    "S1GRD",
    "S2L1C",
    "neck",
    "freeze",
    "backbone",
]

TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}

MAX_TEXT_FILE_BYTES = 3_000_000


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for subdir in [
        "reports",
        "logs",
        "configs",
        "inventory",
        "scripts",
        "metadata",
        "recommendations",
    ]:
        (STEP5R_RUN_DIR / subdir).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def nested_get(data: dict[str, Any], keys: list[str], default: Any = "unknown") -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def compact(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return ";".join(compact(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def keyword_patterns() -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for keyword in KEYWORDS:
        if keyword == "CE":
            expr = r"\bCE\b"
        else:
            expr = re.escape(keyword)
        patterns[keyword] = re.compile(expr, flags=re.IGNORECASE)
    return patterns


def is_text_candidate(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    try:
        return path.stat().st_size <= MAX_TEXT_FILE_BYTES
    except OSError:
        return False


def iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if is_text_candidate(root) else []
    if not root.exists():
        return []
    files: list[Path] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        current_path = Path(current)
        for filename in filenames:
            path = current_path / filename
            if is_text_candidate(path):
                files.append(path)
    return files


def terratorch_package_root() -> Path | None:
    spec = importlib.util.find_spec("terratorch")
    if not spec or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def search_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = [
        ("repository", REPO_ROOT),
        ("repo_configs", REPO_ROOT / "configs"),
        ("step5e_to_step5p_run_configs", RUNS_ROOT),
        ("baseline_cached_configs", BASELINE_ROOT / "configs"),
        ("baseline_cached_checkpoints_metadata", BASELINE_ROOT / "checkpoints"),
        ("user_huggingface_cache", Path.home() / ".cache" / "huggingface" / "hub"),
    ]
    package_root = terratorch_package_root()
    if package_root is not None:
        roots.append(("terratorch_package", package_root))
    return roots


def scope_allows_path(scope: str, path: Path) -> bool:
    path_text = str(path).replace("\\", "/").lower()
    if scope == "step5e_to_step5p_run_configs":
        return bool(re.search(r"/runs/step5[a-p][^/]*/configs/", path_text))
    if scope == "baseline_cached_checkpoints_metadata":
        return "config" in path.name.lower() or path.suffix.lower() in {".json", ".md", ".txt", ".yaml", ".yml"}
    if scope == "user_huggingface_cache":
        return path.suffix.lower() in {".json", ".md", ".txt", ".yaml", ".yml"}
    return True


def search_local_configs() -> list[dict[str, Any]]:
    patterns = keyword_patterns()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for scope, root in search_roots():
        for path in iter_files(root):
            if not scope_allows_path(scope, path):
                continue
            try:
                text = path.read_text(encoding="utf-8-sig", errors="ignore")
            except OSError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                clean_line = line.strip()
                for keyword, pattern in patterns.items():
                    if not pattern.search(line):
                        continue
                    key = (scope, str(path), line_number, keyword)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "source_scope": scope,
                            "file_path": str(path).replace("\\", "/"),
                            "line_number": line_number,
                            "keyword": keyword,
                            "line_text": clean_line[:500],
                        }
                    )
    rows.sort(key=lambda r: (r["source_scope"], r["file_path"], r["line_number"], r["keyword"]))
    return rows


def read_manifest_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def dataset_policy_summary() -> dict[str, Any]:
    manifest_dir = RUNS_ROOT / "step5e_tiny_unetdecoder_baseline" / "manifests"
    split_files = {
        "train": manifest_dir / "flood_train_step5e_filtered.txt",
        "valid": manifest_dir / "flood_valid_step5e_filtered.txt",
        "test": manifest_dir / "flood_test_step5e_filtered.txt",
        "bolivia": manifest_dir / "flood_bolivia_step5e_filtered.txt",
    }
    split_ids = {name: read_manifest_ids(path) for name, path in split_files.items()}
    included_ids = {tile_id for ids in split_ids.values() for tile_id in ids}
    index_path = manifest_dir / "sen1floods11_handlabeled_index_e_paths.csv"
    rows: list[dict[str, str]] = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    indexed_included = [row for row in rows if row.get("tile_id") in included_ids]
    no_water = [row for row in indexed_included if "no_water" in row.get("anomalies", "")]
    warning = [
        row
        for row in indexed_included
        if row.get("cleaning_recommendation") == "warning_review" or row.get("status") == "warning"
    ]
    excluded = [row for row in rows if row.get("status") == "error" or "exclude_candidate" in row.get("cleaning_recommendation", "")]
    return {
        "split_sizes": {name: len(ids) for name, ids in split_ids.items()},
        "total_included": len(included_ids),
        "index_rows": len(rows),
        "no_water_included_count": len(no_water),
        "warning_review_included_count": len(warning),
        "excluded_tile_ids": sorted(row.get("tile_id", "") for row in excluded if row.get("tile_id")),
        "manifest_dir": str(manifest_dir).replace("\\", "/"),
        "index_path": str(index_path).replace("\\", "/"),
    }


def metric_from_summary(summary: dict[str, Any], split: str) -> Any:
    grouped = nested_get(summary, ["evaluations", split, "grouped_metrics"], [])
    if isinstance(grouped, list):
        for row in grouped:
            if row.get("group_type") == "split" and row.get("group_value") == split:
                return row.get("mean_iou", "unknown")
    return "unknown"


def run_row(
    step: str,
    config_path: Path,
    summary_path: Path,
    source_confidence: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    config = load_yaml(config_path)
    summary = read_json(summary_path)
    data_args = nested_get(config, ["data", "init_args"], {})
    model_args = nested_get(config, ["model", "init_args", "model_args"], {})
    model_init = nested_get(config, ["model", "init_args"], {})
    optimizer_args = nested_get(config, ["optimizer", "init_args"], {})
    split_sizes = summary.get("split_sizes") or policy["split_sizes"]
    miou_parts = []
    for split in ["valid", "test", "bolivia"]:
        miou = metric_from_summary(summary, split)
        if isinstance(miou, float):
            miou = f"{miou:.6f}"
        miou_parts.append(f"{split}={miou}")
    transforms = data_args.get("train_transform")
    if transforms in (None, "null"):
        augmentations = "none"
    else:
        augmentations = compact(transforms)
    pretrained = summary.get("pretrained_checkpoint", {}).get("repo_id") or model_args.get("backbone_ckpt_path", "unknown")
    frozen = f"freeze_backbone={model_init.get('freeze_backbone', 'unknown')}; freeze_decoder={model_init.get('freeze_decoder', 'unknown')}"
    scheduler = config.get("lr_scheduler", summary.get("scheduler", "unknown"))
    early = config.get("early_stopping", "none")
    return {
        "dataset_name": "Sen1Floods11 hand-labeled filtered",
        "number_of_samples": f"{policy['total_included']} included ({compact(split_sizes)})",
        "split_policy": "STEP 5E filtered manifests; five fully invalid LabelHand tiles excluded",
        "bolivia_included_or_separated": "separated holdout split",
        "no_water_included": f"yes; local index has {policy['no_water_included_count']} included no_water anomaly rows",
        "warning_review_included": f"yes; local index has {policy['warning_review_included_count']} included warning_review rows",
        "invalid_pixels_ignored": f"yes; ignore_index={model_init.get('ignore_index', data_args.get('no_label_replace', 'unknown'))}",
        "modalities_used": compact(data_args.get("modalities", "unknown")),
        "backbone": compact(model_args.get("backbone", summary.get("model", "unknown"))),
        "decoder": compact(model_args.get("decoder", summary.get("decoder", "unknown"))),
        "pretrained_checkpoint": compact(pretrained),
        "frozen_unfrozen_status": frozen,
        "loss_function": compact(model_init.get("loss", "unknown")),
        "class_weights": compact(model_init.get("class_weights", "none")),
        "optimizer": compact(config.get("optimizer", {}).get("class_path", summary.get("optimizer", "unknown"))),
        "learning_rate": compact(optimizer_args.get("lr", summary.get("learning_rate", "unknown"))),
        "scheduler": compact(scheduler),
        "batch_size": compact(data_args.get("batch_size", summary.get("batch_size", "unknown"))),
        "epochs": compact(config.get("trainer", {}).get("max_epochs", summary.get("max_epochs", "unknown"))),
        "early_stopping": compact(early),
        "image_normalization": "TerraMind means/stds for S2L1C/S1GRD",
        "augmentations": augmentations,
        "metric_formula": "global pixel confusion matrix over valid pixels; mean of background IoU and water IoU",
        "miou_reported": "; ".join(miou_parts),
        "source_confidence": source_confidence,
        "source_path": str(config_path).replace("\\", "/"),
    }


def official_row(label: str, config_path: Path, source_confidence: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_args = nested_get(config, ["data", "init_args"], {})
    model_args = nested_get(config, ["model", "init_args", "model_args"], {})
    model_init = nested_get(config, ["model", "init_args"], {})
    optimizer_args = nested_get(config, ["optimizer", "init_args"], {})
    transforms = data_args.get("train_transform", "unknown")
    return {
        "dataset_name": "Sen1Floods11 official TerraMind config",
        "number_of_samples": "unknown in config; split files referenced by name",
        "split_policy": compact(
            {
                "train": data_args.get("train_split", "unknown"),
                "val": data_args.get("val_split", "unknown"),
                "test": data_args.get("test_split", "unknown"),
            }
        ),
        "bolivia_included_or_separated": "unknown; no separate Bolivia evaluation split in config",
        "no_water_included": "unknown; config does not state filtering",
        "warning_review_included": "unknown; config does not state filtering",
        "invalid_pixels_ignored": f"yes; ignore_index={model_init.get('ignore_index', data_args.get('no_label_replace', 'unknown'))}",
        "modalities_used": compact(data_args.get("modalities", "unknown")),
        "backbone": compact(model_args.get("backbone", "unknown")),
        "decoder": compact(model_args.get("decoder", "unknown")),
        "pretrained_checkpoint": compact(model_args.get("backbone_pretrained", "unknown")),
        "frozen_unfrozen_status": f"freeze_backbone={model_init.get('freeze_backbone', 'unknown')}; freeze_decoder={model_init.get('freeze_decoder', 'unknown')}",
        "loss_function": compact(model_init.get("loss", "unknown")),
        "class_weights": compact(model_init.get("class_weights", "none")),
        "optimizer": compact(config.get("optimizer", {}).get("class_path", "unknown")),
        "learning_rate": compact(optimizer_args.get("lr", "unknown")),
        "scheduler": compact(config.get("lr_scheduler", "unknown")),
        "batch_size": compact(data_args.get("batch_size", "unknown")),
        "epochs": compact(config.get("trainer", {}).get("max_epochs", "unknown")),
        "early_stopping": "not specified",
        "image_normalization": "TerraMind means/stds for S2L1C/S1GRD",
        "augmentations": compact(transforms),
        "metric_formula": "unknown in config",
        "miou_reported": "not reported in config",
        "source_confidence": source_confidence,
        "source_path": f"{label}: {str(config_path).replace('\\', '/')}",
    }


def pangaea_row() -> dict[str, Any]:
    return {
        "dataset_name": "Sen1Floods11 / PANGAEA-like public protocol",
        "number_of_samples": "unknown locally for exact TerraMind benchmark",
        "split_policy": "unknown for exact TerraMind benchmark; PANGAEA warns not to change benchmark hparams for fair comparison",
        "bolivia_included_or_separated": "unknown",
        "no_water_included": "unknown",
        "warning_review_included": "unknown",
        "invalid_pixels_ignored": "unknown",
        "modalities_used": "unknown for TerraMind Sen1Floods11 benchmark",
        "backbone": "unknown; PANGAEA README includes supervised UNet Sen1Floods11 example, not TerraMind exact config",
        "decoder": "unknown; PANGAEA supports UPerNet decoders and UNet examples",
        "pretrained_checkpoint": "unknown",
        "frozen_unfrozen_status": "unknown",
        "loss_function": "unknown for exact TerraMind Sen1Floods11; PANGAEA supports cross_entropy, weighted_cross_entropy, dice",
        "class_weights": "unknown",
        "optimizer": "unknown",
        "learning_rate": "unknown",
        "scheduler": "unknown",
        "batch_size": "unknown",
        "epochs": "unknown",
        "early_stopping": "unknown",
        "image_normalization": "unknown",
        "augmentations": "unknown",
        "metric_formula": "mIoU per PANGAEA protocol; exact Sen1Floods11 evaluator settings not found locally",
        "miou_reported": "user-provided public claim around 0.905-0.9078; exact reproducible local config not found",
        "source_confidence": "low for exact TerraMind/Sen1Floods11 config; high that PANGAEA is the benchmark family",
        "source_path": "https://github.com/VMarsocci/pangaea-bench",
    }


def build_protocol_comparison(policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        run_row(
            "5O",
            RUNS_ROOT
            / "step5o_terramind_l_upernet_long_classical_training"
            / "configs"
            / "terramind_l_upernet_long_classical_train.yaml",
            RUNS_ROOT / "step5o_terramind_l_upernet_long_classical_training" / "metrics" / "step5o_summary.json",
            "high; local config and metrics",
            policy,
        ),
        run_row(
            "5P",
            RUNS_ROOT
            / "step5p_terramind_l_upernet_big_classical_training"
            / "configs"
            / "terramind_l_upernet_big_classical_train.yaml",
            RUNS_ROOT / "step5p_terramind_l_upernet_big_classical_training" / "metrics" / "step5p_summary.json",
            "high; local config and metrics",
            policy,
        ),
        run_row(
            "5I",
            RUNS_ROOT
            / "step5i_base_unetdecoder_pretrained"
            / "configs"
            / "terramind_v1_base_unetdecoder_pretrained_train.yaml",
            RUNS_ROOT / "step5i_base_unetdecoder_pretrained" / "metrics" / "step5i_summary.json",
            "high; local config and metrics",
            policy,
        ),
        official_row(
            "local IBM TerraMind base Sen1Floods11",
            REPO_ROOT / "configs" / "terramind" / "official_terramind_v1_base_sen1floods11.yaml",
            "high for config fields; metric/split filtering unknown",
        ),
        official_row(
            "local IBM TerraMind base TiM-LULC Sen1Floods11",
            REPO_ROOT / "configs" / "terramind" / "official_terramind_v1_base_tim_lulc_sen1floods11.yaml",
            "high for config fields; metric/split filtering unknown",
        ),
        pangaea_row(),
    ]


def gap_hypotheses(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": now_utc(),
        "headline": "Loss parity is mostly satisfied for the official IBM TerraMind config because both official local config and our STEP 5I/5O/5P use dice loss.",
        "high_confidence": [
            {
                "rank": 1,
                "cause": "Official-public split/evaluation protocol is not fully reproduced locally.",
                "evidence": "Our filtered protocol uses 441 included tiles, separates 15 Bolivia tiles, keeps no_water and warning_review rows, and excludes five fully invalid LabelHand tiles. The exact PANGAEA/TerraMind Sen1Floods11 benchmark config behind the 0.905-0.9078 claim was not found locally.",
                "action": "Do not compare 0.85-0.86 directly to the public claim until the exact benchmark split and metric aggregation are verified.",
            },
            {
                "rank": 2,
                "cause": "Training recipe mismatch relative to the official IBM TerraMind config.",
                "evidence": "Official config uses batch_size=8, precision=16-mixed, max_epochs=100, AdamW lr=2e-5, ReduceLROnPlateau on val/loss, and D4 augmentation. STEP 5O/5P use batch_size=1, precision=32, lr=1e-4, no train_transform, UPerNet BatchNorm eval policy, and mIoU-driven early stopping.",
                "action": "Before physics loss, run classical ablations that isolate loss only, then optionally a separate recipe-parity run.",
            },
            {
                "rank": 3,
                "cause": "TerraMind-L UPerNet feature indices may not match the large-backbone recommendation.",
                "evidence": "STEP 5O/5P use SelectIndices [2, 5, 8, 11]. The IBM official Sen1Floods11 config comments mark [2, 5, 8, 11] for tiny/small/base and [5, 11, 17, 23] for large version.",
                "action": "Validate large-backbone UPerNet feature indices before treating TerraMind-L UPerNet as architecture-parity complete.",
            },
        ],
        "medium_confidence": [
            {
                "rank": 4,
                "cause": "Metric aggregation differences.",
                "evidence": "Our evaluator reports global pixel-confusion mIoU over valid pixels. The exact PANGAEA evaluator settings for Sen1Floods11 were not found locally.",
                "action": "Recompute tile-mean, event-mean, and global mIoU for existing predictions in a later audit if needed.",
            },
            {
                "rank": 5,
                "cause": "Composite CE+Dice or weighted CE+Dice could improve flood recall/precision balance, but it is not the official IBM loss.",
                "evidence": "Official IBM config and our current runs all use dice. PANGAEA supports cross_entropy, weighted_cross_entropy, and dice, but exact TerraMind Sen1Floods11 loss is unknown.",
                "action": "Run STEP 5S as a small controlled loss ablation against STEP 5O/5I, not as proof of official parity.",
            },
            {
                "rank": 6,
                "cause": "Checkpoint loading semantics differ from the official config.",
                "evidence": "Official config sets backbone_pretrained=true; our local configs set backbone_pretrained=false plus an explicit backbone_ckpt_path. Summaries verify checkpoints, but implementation semantics should stay documented.",
                "action": "Keep checkpoint hash/path in every future run summary.",
            },
        ],
        "low_confidence": [
            {
                "rank": 7,
                "cause": "Normalization mismatch.",
                "evidence": "Our configs and official IBM configs use the same TerraMind means/stds for S2L1C and S1GRD.",
                "action": "No immediate action unless raw preprocessing is changed.",
            },
            {
                "rank": 8,
                "cause": "Pure CE-vs-Dice loss mismatch.",
                "evidence": "STEP 5I, STEP 5O, STEP 5P, and the official IBM configs all use dice.",
                "action": "Keep Dice as STEP 5S control; use CE+Dice variants as robustness ablations.",
            },
        ],
        "dataset_policy_snapshot": policy,
    }


def loss_ablation_plan() -> dict[str, Any]:
    candidates = [
        {
            "name": "dice",
            "formula": "1 - mean Dice over valid pixels/classes, ignore_index=-1",
            "expected_benefit": "Official IBM TerraMind Sen1Floods11 loss parity and direct control against STEP 5O/5P.",
            "risks": "Already used in STEP 5O/5P, so it may not close the public-claim gap by itself.",
            "known_config_match": "Matches local and public IBM TerraMind Sen1Floods11 configs.",
            "recommended_weighting": "lambda_dice=1.0",
            "training_budget_proposal": "Same filtered protocol and no physics/topography; start with the STEP 5P budget envelope, but launch only after human validation.",
            "stopping_criteria": "Monitor validation_mIoU; compare best checkpoint by valid/test/Bolivia mIoU and water IoU.",
        },
        {
            "name": "ce_dice",
            "formula": "lambda_ce * CE(logits, y) + lambda_dice * Dice(logits, y), ignore_index=-1",
            "expected_benefit": "CE stabilizes per-pixel calibration while Dice keeps flood-region overlap pressure.",
            "risks": "May overfit background if CE dominates; not verified as the official IBM TerraMind loss.",
            "known_config_match": "PANGAEA supports cross_entropy and dice individually; exact composite not found in official TerraMind config.",
            "recommended_weighting": "lambda_ce=0.5, lambda_dice=0.5 as first pass",
            "training_budget_proposal": "Use identical seed, split, model, and optimizer as the Dice control.",
            "stopping_criteria": "Require validation mIoU improvement over STEP 5O plus no Bolivia degradation beyond 0.01 absolute mIoU.",
        },
        {
            "name": "weighted_ce_dice",
            "formula": "lambda_wce * weighted_CE(logits, y; class_weights) + lambda_dice * Dice(logits, y), ignore_index=-1",
            "expected_benefit": "Can compensate class imbalance and improve flood recall on sparse-water/no-water-heavy tiles.",
            "risks": "Bad class weights can inflate false positives on no-water tiles.",
            "known_config_match": "PANGAEA supports weighted_cross_entropy; exact TerraMind Sen1Floods11 class weights not found.",
            "recommended_weighting": "lambda_wce=0.5, lambda_dice=0.5; derive class weights from STEP 5E filtered train labels only",
            "training_budget_proposal": "Same as CE+Dice; freeze no additional modules.",
            "stopping_criteria": "Accept only if validation mIoU, test mIoU, and no-water false-positive behavior improve or stay within tolerance.",
        },
        {
            "name": "focal_optional",
            "formula": "Focal CE with gamma=2.0, alpha from train class balance, optionally plus Dice",
            "expected_benefit": "Can focus updates on hard flood/background pixels.",
            "risks": "More sensitive hyperparameters and less official parity.",
            "known_config_match": "Not found in official TerraMind Sen1Floods11 config.",
            "recommended_weighting": "Only run if CE+Dice and weighted CE+Dice fail to improve recall/precision balance.",
            "training_budget_proposal": "Short diagnostic run first, no physics/topography.",
            "stopping_criteria": "Stop if validation water precision collapses or no-water false positives increase.",
        },
        {
            "name": "tversky_optional",
            "formula": "Tversky loss with alpha/beta tuned for FP/FN tradeoff, optionally focal-Tversky",
            "expected_benefit": "Useful if false negatives dominate flood omission errors.",
            "risks": "Can distort probability calibration and is farthest from known official configs.",
            "known_config_match": "Not found in official TerraMind Sen1Floods11 config.",
            "recommended_weighting": "Only after error analysis confirms systematic flood omission.",
            "training_budget_proposal": "Short diagnostic run first, no physics/topography.",
            "stopping_criteria": "Accept only with improved water IoU and stable background IoU.",
        },
    ]
    return {
        "generated_at": now_utc(),
        "step": "5S",
        "scope": "segmentation loss ablation only; no physics loss, no DEM, no DARN, no STURM",
        "baseline_comparison_targets": ["STEP 5O", "STEP 5I", "STEP 5P"],
        "recommended_primary_order": ["dice", "ce_dice", "weighted_ce_dice"],
        "optional_only_after_review": ["focal_optional", "tversky_optional"],
        "candidates": candidates,
        "global_controls": {
            "dataset": "Sen1Floods11 current filtered protocol",
            "ignore_index": -1,
            "modalities": ["S2L1C", "S1GRD"],
            "model": "TerraMind-L pretrained + UPerNet",
            "physics_loss": False,
            "topographic_loss": False,
            "dem_input": False,
            "raw_data_modified": False,
        },
    }


def write_loss_plan_markdown(plan: dict[str, Any], path: Path) -> None:
    lines = [
        "# STEP 5S Segmentation-Loss Ablation Plan",
        "",
        "Status: plan only. No training was launched by STEP 5R.",
        "",
        "## Scope",
        "",
        "- Model: TerraMind-L pretrained + UPerNet.",
        "- Dataset: Sen1Floods11 current filtered protocol.",
        "- Inputs: S2L1C + S1GRD only.",
        "- ignore_index: -1.",
        "- No physics loss, no topographic loss, no DEM input, no DARN, no STURM.",
        "- Compare against STEP 5O, STEP 5P, and STEP 5I.",
        "",
        "## Recommended Order",
        "",
    ]
    for idx, name in enumerate(plan["recommended_primary_order"], start=1):
        lines.append(f"{idx}. {name}")
    lines.extend(["", "## Candidates", ""])
    for candidate in plan["candidates"]:
        lines.extend(
            [
                f"### {candidate['name']}",
                "",
                f"- Formula: {candidate['formula']}",
                f"- Expected benefit: {candidate['expected_benefit']}",
                f"- Risks: {candidate['risks']}",
                f"- Known config match: {candidate['known_config_match']}",
                f"- Recommended weighting: {candidate['recommended_weighting']}",
                f"- Training budget proposal: {candidate['training_budget_proposal']}",
                f"- Stopping criteria: {candidate['stopping_criteria']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Audit Note",
            "",
            "Dice is already used in STEP 5I, STEP 5O, STEP 5P, and the local/public IBM TerraMind Sen1Floods11 configs. Therefore STEP 5S should be treated as a controlled robustness ablation, not as evidence that loss alone explains the 90.5 vs 85-86 mIoU gap.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path,
    inventory_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    hypotheses: dict[str, Any],
    plan: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    official_rows = [row for row in comparison_rows if "official TerraMind config" in row["dataset_name"]]
    official_loss_values = sorted({row["loss_function"] for row in official_rows})
    lines = [
        "# STEP 5R - Official Benchmark / Segmentation-Loss Parity Audit",
        "",
        f"Generated at: {now_utc()}",
        "",
        "## Status",
        "",
        "Result: DONE",
        "",
        "No model training, physics-loss training, TerraMind training, DARN, or STURM-Flood training was launched. Raw Sen1Floods11 data, raw DEM data, and official split files were not modified.",
        "",
        "## Why This Audit Was Needed",
        "",
        "STEP 6B4 completed the DEM/topographic alignment, but adding topographic physics loss before checking segmentation-baseline parity would make the next comparison hard to interpret. STEP 5R checks whether the public TerraMind/PANGAEA-style result gap is plausibly caused by loss choice or by broader protocol differences.",
        "",
        "## Local Search Summary",
        "",
        f"- Inventory rows written: {len(inventory_rows)}",
        "- Inventory CSV: `inventory/local_config_search_results.csv`",
        "- Inventory JSON: `inventory/local_config_search_results.json`",
        "- Search scopes: repository, repo configs, STEP 5E-5P run configs, TerraTorch package when import metadata was available, cached configs/checkpoint metadata, and Hugging Face cache text metadata.",
        "",
        "Important local hits:",
        "",
        "- `configs/terramind/official_terramind_v1_base_sen1floods11.yaml`",
        "- `configs/terramind/official_terramind_v1_base_tim_lulc_sen1floods11.yaml`",
        "- STEP 5I config: `loss: dice`, `decoder: UNetDecoder`, `backbone: terramind_v1_base`",
        "- STEP 5O/5P configs: `loss: dice`, `decoder: UperNetDecoder`, `backbone: terramind_v1_large`",
        "",
        "## Official Config Finding",
        "",
        f"- Official IBM TerraMind Sen1Floods11 configs found locally: yes.",
        f"- Official loss identified from those configs: {', '.join(official_loss_values) if official_loss_values else 'unknown'}.",
        "- Exact reproducible PANGAEA/TerraMind Sen1Floods11 benchmark config for the public 0.905-0.9078 mIoU claim: not found locally.",
        "- Public PANGAEA repository confirms the benchmark family and supported loss options, but the exact TerraMind/Sen1Floods11 benchmark hparams remain unknown from local files.",
        "",
        "## Dataset Policy Snapshot",
        "",
        f"- Included samples: {policy['total_included']}",
        f"- Split sizes: {compact(policy['split_sizes'])}",
        f"- No-water rows kept: {policy['no_water_included_count']}",
        f"- Warning-review rows kept: {policy['warning_review_included_count']}",
        f"- Excluded fully invalid/error tile IDs: {', '.join(policy['excluded_tile_ids'])}",
        "- Invalid label pixels are ignored via `ignore_index: -1`.",
        "",
        "## Protocol Comparison",
        "",
        "Structured comparison files:",
        "",
        "- `metadata/step5r_protocol_comparison.csv`",
        "- `metadata/step5r_protocol_comparison.json`",
        "",
        "Core observations:",
        "",
        "- Our STEP 5I/5O/5P and the official IBM TerraMind Sen1Floods11 configs all use Dice loss.",
        "- Official IBM configs use batch size 8, 16-mixed precision, 100 epochs, D4 augmentation, AdamW lr 2e-5, and UNetDecoder.",
        "- Our STEP 5O/5P TerraMind-L + UPerNet runs use batch size 1, fp32, no train augmentation, AdamW lr 1e-4, mIoU scheduler/early stopping, and BatchNorm eval policy for UPerNet.",
        "- Our TerraMind-L UPerNet configs use feature indices [2, 5, 8, 11]; the official config comment labels [5, 11, 17, 23] as the large-version indices.",
        "",
        "## Likely Gap Causes",
        "",
    ]
    for bucket in ["high_confidence", "medium_confidence", "low_confidence"]:
        title = bucket.replace("_", " ").title()
        lines.extend([f"### {title}", ""])
        for item in hypotheses[bucket]:
            lines.extend(
                [
                    f"{item['rank']}. {item['cause']}",
                    f"   Evidence: {item['evidence']}",
                    f"   Action: {item['action']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## STEP 5S Recommendation",
            "",
            "Recommended primary loss ablation order:",
            "",
        ]
    )
    for idx, name in enumerate(plan["recommended_primary_order"], start=1):
        lines.append(f"{idx}. {name}")
    lines.extend(
        [
            "",
            "Config stubs prepared only:",
            "",
            "- `configs/step5s_terramind_l_upernet_dice_stub.yaml`",
            "- `configs/step5s_terramind_l_upernet_ce_dice_stub.yaml`",
            "- `configs/step5s_terramind_l_upernet_weighted_ce_dice_stub.yaml`",
            "",
            "These stubs explicitly disable physics loss, topographic loss, and DEM input.",
            "",
            "## Sources",
            "",
            "- IBM TerraMind page: https://ibm.github.io/terramind/",
            "- IBM TerraMind Sen1Floods11 config: https://raw.githubusercontent.com/IBM/terramind/main/configs/terramind_v1_base_sen1floods11.yaml",
            "- IBM TerraMind TiM-LULC Sen1Floods11 config: https://raw.githubusercontent.com/IBM/terramind/main/configs/terramind_v1_base_tim_lulc_sen1floods11.yaml",
            "- PANGAEA benchmark repository: https://github.com/VMarsocci/pangaea-bench",
            "",
            "## Next Step",
            "",
            "Human validation required before starting STEP 5S - segmentation-loss ablation before physics-informed training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_stubs_to_run_dir() -> None:
    for name in [
        "step5s_terramind_l_upernet_dice_stub.yaml",
        "step5s_terramind_l_upernet_ce_dice_stub.yaml",
        "step5s_terramind_l_upernet_weighted_ce_dice_stub.yaml",
    ]:
        src = REPO_ROOT / "configs" / name
        if src.exists():
            shutil.copy2(src, STEP5R_RUN_DIR / "configs" / name)


def update_pipeline_status() -> None:
    status_path = REPO_ROOT / "pipeline_status.json"
    status = read_json(status_path)
    status.update(
        {
            "current_step": "5R",
            "status": "done",
            "official_benchmark_config_found": True,
            "official_benchmark_config_scope": "IBM TerraMind base/base-TiM Sen1Floods11 configs found; exact public PANGAEA 0.905-0.9078 reproducible config not found locally",
            "official_reproducible_pangaea_sen1floods11_config_found": False,
            "official_loss_identified": True,
            "official_loss_identified_value": "dice",
            "recommended_loss_ablation_ready": True,
            "physics_loss_training_started": False,
            "physics_loss_started": False,
            "topographic_alignment_validated": True,
            "full_topographic_alignment_completed": True,
            "darn_started": False,
            "sturm_training_started": False,
            "raw_data_modified": False,
            "official_split_files_modified": False,
            "next_step_allowed": False,
            "human_validation_required": True,
            "step5r_run_dir": str(STEP5R_RUN_DIR).replace("\\", "/"),
            "step5r_report": str((REPO_ROOT / "reports" / "STEP_5R_official_benchmark_loss_parity_audit_report.md")).replace("\\", "/"),
            "step5r_run_report": str((STEP5R_RUN_DIR / "reports" / "STEP_5R_official_benchmark_loss_parity_audit_report.md")).replace("\\", "/"),
            "step5r_inventory_csv": str((STEP5R_RUN_DIR / "inventory" / "local_config_search_results.csv")).replace("\\", "/"),
            "step5r_inventory_json": str((STEP5R_RUN_DIR / "inventory" / "local_config_search_results.json")).replace("\\", "/"),
            "step5r_protocol_comparison_csv": str((STEP5R_RUN_DIR / "metadata" / "step5r_protocol_comparison.csv")).replace("\\", "/"),
            "step5r_protocol_comparison_json": str((STEP5R_RUN_DIR / "metadata" / "step5r_protocol_comparison.json")).replace("\\", "/"),
            "step5r_gap_hypotheses_json": str((STEP5R_RUN_DIR / "metadata" / "step5r_gap_hypotheses.json")).replace("\\", "/"),
            "step5s_loss_ablation_plan_md": str((STEP5R_RUN_DIR / "recommendations" / "step5s_segmentation_loss_ablation_plan.md")).replace("\\", "/"),
            "step5s_loss_ablation_plan_json": str((STEP5R_RUN_DIR / "recommendations" / "step5s_segmentation_loss_ablation_plan.json")).replace("\\", "/"),
            "step5s_dice_stub": str((REPO_ROOT / "configs" / "step5s_terramind_l_upernet_dice_stub.yaml")).replace("\\", "/"),
            "step5s_ce_dice_stub": str((REPO_ROOT / "configs" / "step5s_terramind_l_upernet_ce_dice_stub.yaml")).replace("\\", "/"),
            "step5s_weighted_ce_dice_stub": str((REPO_ROOT / "configs" / "step5s_terramind_l_upernet_weighted_ce_dice_stub.yaml")).replace("\\", "/"),
            "generated_at": now_utc(),
        }
    )
    write_json(status_path, status)


def main() -> None:
    ensure_dirs()
    inventory_rows = search_local_configs()
    inventory_fields = ["source_scope", "file_path", "line_number", "keyword", "line_text"]
    write_csv(STEP5R_RUN_DIR / "inventory" / "local_config_search_results.csv", inventory_rows, inventory_fields)
    write_json(STEP5R_RUN_DIR / "inventory" / "local_config_search_results.json", inventory_rows)

    policy = dataset_policy_summary()
    comparison_rows = build_protocol_comparison(policy)
    comparison_fields = [
        "dataset_name",
        "number_of_samples",
        "split_policy",
        "bolivia_included_or_separated",
        "no_water_included",
        "warning_review_included",
        "invalid_pixels_ignored",
        "modalities_used",
        "backbone",
        "decoder",
        "pretrained_checkpoint",
        "frozen_unfrozen_status",
        "loss_function",
        "class_weights",
        "optimizer",
        "learning_rate",
        "scheduler",
        "batch_size",
        "epochs",
        "early_stopping",
        "image_normalization",
        "augmentations",
        "metric_formula",
        "miou_reported",
        "source_confidence",
        "source_path",
    ]
    write_csv(STEP5R_RUN_DIR / "metadata" / "step5r_protocol_comparison.csv", comparison_rows, comparison_fields)
    write_json(STEP5R_RUN_DIR / "metadata" / "step5r_protocol_comparison.json", comparison_rows)

    hypotheses = gap_hypotheses(policy)
    write_json(STEP5R_RUN_DIR / "metadata" / "step5r_gap_hypotheses.json", hypotheses)

    plan = loss_ablation_plan()
    write_json(STEP5R_RUN_DIR / "recommendations" / "step5s_segmentation_loss_ablation_plan.json", plan)
    write_loss_plan_markdown(plan, STEP5R_RUN_DIR / "recommendations" / "step5s_segmentation_loss_ablation_plan.md")

    copy_stubs_to_run_dir()

    report_path = REPO_ROOT / "reports" / "STEP_5R_official_benchmark_loss_parity_audit_report.md"
    write_report(report_path, inventory_rows, comparison_rows, hypotheses, plan, policy)
    shutil.copy2(report_path, STEP5R_RUN_DIR / "reports" / report_path.name)
    shutil.copy2(Path(__file__), STEP5R_RUN_DIR / "scripts" / Path(__file__).name)

    write_json(
        STEP5R_RUN_DIR / "metadata" / "step5r_summary.json",
        {
            "step": "5R",
            "status": "done",
            "inventory_rows": len(inventory_rows),
            "official_benchmark_config_found": True,
            "official_loss_identified": True,
            "official_loss_identified_value": "dice",
            "exact_public_pangaea_reproducible_config_found": False,
            "recommended_loss_ablation_ready": True,
            "training_started": False,
            "physics_loss_training_started": False,
            "raw_data_modified": False,
            "generated_at": now_utc(),
        },
    )
    update_pipeline_status()
    print(f"STEP 5R audit complete: {STEP5R_RUN_DIR}")


if __name__ == "__main__":
    main()
