from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from losses.combined_loss import CombinedSegmentationPhysicsLoss  # noqa: E402
from losses.physics_topographic_loss import TopographicInconsistencyLoss  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b4_full_dem_alignment"
)
DEFAULT_MANIFEST = DEFAULT_RUN_DIR / "manifests" / "topography_full_manifest.csv"
DEFAULT_SUMMARY = DEFAULT_RUN_DIR / "metrics" / "step6b4_loss_compatibility_smoke_summary.json"
EXPECTED_TOTAL = 441
EXPECTED_SPLIT_COUNTS = {"train": 251, "valid": 86, "test": 89, "bolivia": 15}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_prop(obj: dict[str, Any], key: str, value: Any) -> None:
    obj[key] = value


def load_real_sample(row: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    with rasterio.open(row["label_path"]) as label_ds:
        label = label_ds.read(1).astype("int64")
    with rasterio.open(row["topography_path"]) as topo_ds:
        topography = topo_ds.read(1).astype("float32")

    target_np = np.where((label == 0) | (label == 1), label, -1).astype("int64")
    target = torch.from_numpy(target_np)[None]
    topo = torch.from_numpy(topography)[None]
    finite_ratio = float(np.isfinite(topography).sum() / topography.size)
    valid_ratio = float(((target_np == 0) | (target_np == 1)).sum() / target_np.size)
    return target, topo, {"finite_ratio": finite_ratio, "label_valid_ratio": valid_ratio}


def select_smoke_rows(rows: list[dict[str, str]], seed: int, max_samples: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    ok_rows = [row for row in rows if row.get("status") == "ok" and Path(row["topography_path"]).exists()]
    selected: list[dict[str, str]] = []
    seen_tile_ids: set[str] = set()

    for split in EXPECTED_SPLIT_COUNTS:
        candidates = sorted([row for row in ok_rows if row["split"] == split], key=lambda item: item["tile_id"])
        for row in rng.sample(candidates, min(2, len(candidates))):
            if row["tile_id"] not in seen_tile_ids:
                selected.append(row)
                seen_tile_ids.add(row["tile_id"])

    for event_location in sorted({row["event_location"] for row in ok_rows}):
        if len(selected) >= max_samples:
            break
        candidates = sorted(
            [row for row in ok_rows if row["event_location"] == event_location and row["tile_id"] not in seen_tile_ids],
            key=lambda item: item["tile_id"],
        )
        if candidates:
            row = rng.choice(candidates)
            selected.append(row)
            seen_tile_ids.add(row["tile_id"])

    return selected[:max_samples]


def run_one(row: dict[str, str], seed: int) -> dict[str, Any]:
    target, topography, sample_stats = load_real_sample(row)
    height, width = int(target.shape[-2]), int(target.shape[-1])
    torch.manual_seed(seed)

    logits_topo = torch.randn(1, 2, height, width, dtype=torch.float32, requires_grad=True)
    topo_loss_fn = TopographicInconsistencyLoss(neighborhood="4", elevation_scale=1.0)
    loss_topo = topo_loss_fn(logits=logits_topo, target=target, topography=topography)
    loss_topo.backward()
    topo_gradient_l1 = float(logits_topo.grad.detach().abs().sum())

    logits_combined = torch.randn(1, 2, height, width, dtype=torch.float32, requires_grad=True)
    combined_loss_fn = CombinedSegmentationPhysicsLoss(lambda_topo=0.05, neighborhood="4")
    combined = combined_loss_fn(logits=logits_combined, target=target, topography=topography)
    combined["loss_total"].backward()
    combined_gradient_l1 = float(logits_combined.grad.detach().abs().sum())

    finite_losses = bool(torch.isfinite(loss_topo).item() and torch.isfinite(combined["loss_total"]).item())
    gradients_ok = topo_gradient_l1 > 0.0 and combined_gradient_l1 > 0.0
    return {
        "tile_id": row["tile_id"],
        "split": row["split"],
        "event_location": row["event_location"],
        "label_path": row["label_path"],
        "topography_path": row["topography_path"],
        "logits_shape": [1, 2, height, width],
        "loss_topo": float(loss_topo.detach()),
        "loss_total": float(combined["loss_total"].detach()),
        "loss_seg": float(combined["loss_seg"].detach()),
        "combined_loss_topo": float(combined["loss_topo"].detach()),
        "lambda_topo": float(combined["lambda_topo"].detach()),
        "topo_gradient_l1": topo_gradient_l1,
        "combined_gradient_l1": combined_gradient_l1,
        "finite_ratio": sample_stats["finite_ratio"],
        "label_valid_ratio": sample_stats["label_valid_ratio"],
        "finite_losses": finite_losses,
        "gradients_ok": gradients_ok,
        "status": "passed" if finite_losses and gradients_ok else "failed",
    }


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_config_stubs(repo_root: Path, manifest_path: Path) -> tuple[Path, Path]:
    config_dir = repo_root / "configs" / "physics_loss"
    config_dir.mkdir(parents=True, exist_ok=True)
    main_path = config_dir / "terramind_l_upernet_topographic_loss_ready_manifest_stub.yaml"
    control_path = config_dir / "terramind_base_unetdecoder_topographic_loss_control_ready_manifest_stub.yaml"
    common = f"""topography:
  manifest_path: "{manifest_path.as_posix()}"
  topography_type: dem_copernicus_glo30
  source: Copernicus DEM GLO-30
  source_limitation: "DSM-like elevation product, not HAND or guaranteed bare-earth DTM"
physics_loss:
  lambda_candidates:
    - 0.01
    - 0.05
    - 0.1
  ignore_index: -1
  status: ready_for_human_review_before_physics_training
guards:
  full_topographic_alignment_required: true
  human_validation_required: true
  physics_loss_training_started: false
"""
    main_path.write_text(
        """model:
  name: TerraMind-L pretrained + UPerNet
  classical_baseline_step: STEP 5O
"""
        + common,
        encoding="utf-8",
    )
    control_path.write_text(
        """model:
  name: TerraMind base pretrained + UNetDecoder
  classical_baseline_step: STEP 5I
  role: control_physics_baseline
"""
        + common,
        encoding="utf-8",
    )
    return main_path, control_path


def write_report(
    *,
    repo_root: Path,
    run_dir: Path,
    alignment_summary: dict[str, Any],
    loss_summary: dict[str, Any],
    manifest_path: Path,
    main_config: Path,
    control_config: Path,
) -> tuple[Path, Path]:
    report_path = repo_root / "reports" / "STEP_6B4_full_dem_alignment_report.md"
    run_report_path = run_dir / "reports" / "STEP_6B4_full_dem_alignment_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run_report_path.parent.mkdir(parents=True, exist_ok=True)

    figures = sorted((run_dir / "figures").glob("*.png"))
    split_counts = alignment_summary.get("split_level_counts", {})
    event_counts = alignment_summary.get("event_location_level_counts", {})
    figure_lines = "\n".join(f"- `figures/{path.name}`" for path in figures)
    split_lines = "\n".join(
        f"- {split}: selected {values.get('selected')}, passed {values.get('passed')}, expected {values.get('expected')}"
        for split, values in split_counts.items()
    )
    event_lines = "\n".join(
        f"- {event}: selected {values.get('selected')}, passed {values.get('passed')}"
        for event, values in event_counts.items()
    )

    text = f"""# STEP 6B4 - Full Copernicus DEM Alignment

Generated: 2026-06-21

## Scope

STEP 6B4 aligned Copernicus DEM GLO-30 to all valid Sen1Floods11 hand-labeled `LabelHand` grids. No model training, physics-loss training, TerraMind training, DARN, or STURM-Flood training was started. Raw Sen1Floods11 data, raw DEM files, and official split files were not modified.

## DEM Source

- Source: Copernicus DEM GLO-30
- Local source folder: `E:/flood_research/data/raw/dem/copernicus_glo30/`
- Verified DEM tiles from STEP 6B2b: 53 / 53
- Output folder: `E:/flood_research/data/derived/sen1floods11_topography/dem_aligned/`

## Alignment Method

Each verified Sen1Floods11 sample was matched to an intersecting Copernicus DEM tile. The DEM was bilinearly reprojected/resampled to the exact `LabelHand` CRS, transform, and `512x512` grid. Existing valid outputs are reused unless `--overwrite` is requested.

## Full QC Summary

- Expected samples: {alignment_summary.get('expected_samples')}
- Input selected samples: {alignment_summary.get('input_selected_samples')}
- Aligned samples: {alignment_summary.get('aligned_samples')}
- Valid outputs: {alignment_summary.get('valid_outputs')}
- Missing outputs: {alignment_summary.get('missing_outputs')}
- Failed outputs: {alignment_summary.get('failed_outputs')}
- Shape pass count: {alignment_summary.get('shape_pass_count')}
- CRS pass count: {alignment_summary.get('crs_pass_count')}
- Transform pass count: {alignment_summary.get('transform_pass_count')}
- Finite ratio min: {alignment_summary.get('finite_ratio_min')}
- Finite ratio mean: {alignment_summary.get('finite_ratio_mean')}
- Corrupted count: {alignment_summary.get('corrupted_count')}

## Split Counts

{split_lines}

## Event Counts

{event_lines}

## Manifests And Metrics

- Full manifest CSV: `{manifest_path.as_posix()}`
- Full manifest JSON: `{(run_dir / 'manifests' / 'topography_full_manifest.json').as_posix()}`
- Full QC CSV: `{(run_dir / 'metrics' / 'step6b4_full_alignment_qc.csv').as_posix()}`
- Full QC JSON: `{(run_dir / 'metrics' / 'step6b4_full_alignment_qc.json').as_posix()}`
- Full summary JSON: `{(run_dir / 'metrics' / 'step6b4_full_alignment_summary.json').as_posix()}`

## QC Figures

{figure_lines}

## Loss Compatibility Smoke

- Status: {loss_summary.get('status')}
- Tested samples: {loss_summary.get('tested_samples')}
- Passed samples: {loss_summary.get('passed_samples')}
- Failed samples: {loss_summary.get('failed_samples')}
- Uses real aligned topography: true
- Uses synthetic logits: true
- Summary: `{(run_dir / 'metrics' / 'step6b4_loss_compatibility_smoke_summary.json').as_posix()}`

## Training-Ready Manifest Stubs

- Main target config stub: `{main_config.as_posix()}`
- Control config stub: `{control_config.as_posix()}`

## Limitations

- Copernicus DEM GLO-30 is DSM-like elevation, not HAND and not guaranteed bare-earth DTM.
- Buildings and vegetation may affect local monotonic assumptions.
- DEM is about 30 m resolution while Sen1Floods11 chips are `512x512` in `EPSG:4326`.
- Bilinear resampling can smooth local relief.
- This validates topographic raster alignment, not hydraulic correctness.

## Result

Full topographic alignment passed: {alignment_summary.get('full_topographic_alignment_qc_passed')}

Physics-informed training can be considered next only after human validation of this report, the QC figures, and the manifest stubs.

## Next Step

Human validation is required before STEP 6C: first physics-informed training on TerraMind-L + UPerNet.
"""
    report_path.write_text(text, encoding="utf-8")
    run_report_path.write_text(text, encoding="utf-8")
    return report_path, run_report_path


def update_pipeline_status(
    *,
    repo_root: Path,
    run_dir: Path,
    alignment_summary: dict[str, Any],
    loss_summary: dict[str, Any],
    report_path: Path,
    run_report_path: Path,
    main_config: Path,
    control_config: Path,
) -> None:
    status_path = repo_root / "pipeline_status.json"
    status = read_json_if_exists(status_path)
    alignment_ok = bool(alignment_summary.get("full_topographic_alignment_qc_passed"))
    smoke_ok = bool(loss_summary.get("loss_smoke_passed"))
    validated = (
        alignment_ok
        and smoke_ok
        and alignment_summary.get("expected_samples") == EXPECTED_TOTAL
        and alignment_summary.get("aligned_samples") == EXPECTED_TOTAL
        and alignment_summary.get("valid_outputs") == EXPECTED_TOTAL
    )

    set_prop(status, "current_step", "6B4")
    set_prop(status, "status", "done" if validated else "blocked")
    set_prop(status, "dem_source_available", True)
    set_prop(status, "copernicus_dem_verified", True)
    set_prop(status, "sample_topographic_alignment_completed", True)
    set_prop(status, "sample_topographic_alignment_qc_passed", True)
    set_prop(status, "full_topographic_alignment_started", True)
    set_prop(status, "full_topographic_alignment_completed", alignment_summary.get("aligned_samples") == EXPECTED_TOTAL)
    set_prop(status, "full_topographic_alignment_qc_passed", alignment_ok)
    set_prop(status, "topography_full_manifest_created", alignment_summary.get("topography_full_manifest_created", False))
    set_prop(status, "topographic_loss_compatibility_smoke_passed", smoke_ok)
    set_prop(status, "topographic_alignment_validated", validated)
    set_prop(status, "physics_loss_training_started", False)
    set_prop(status, "physics_loss_started", False)
    set_prop(status, "darn_started", False)
    set_prop(status, "sturm_started", False)
    set_prop(status, "sturm_training_started", False)
    set_prop(status, "sturm_flood_started", False)
    set_prop(status, "raw_data_modified", False)
    set_prop(status, "official_split_files_modified", False)
    set_prop(status, "next_step_allowed", False)
    set_prop(status, "human_validation_required", True)
    set_prop(status, "step6b4_run_dir", run_dir.as_posix())
    set_prop(status, "step6b4_report", report_path.as_posix())
    set_prop(status, "step6b4_run_report", run_report_path.as_posix())
    set_prop(status, "step6b4_topography_full_manifest_csv", (run_dir / "manifests" / "topography_full_manifest.csv").as_posix())
    set_prop(status, "step6b4_topography_full_manifest_json", (run_dir / "manifests" / "topography_full_manifest.json").as_posix())
    set_prop(status, "step6b4_full_alignment_qc_csv", (run_dir / "metrics" / "step6b4_full_alignment_qc.csv").as_posix())
    set_prop(status, "step6b4_full_alignment_qc_json", (run_dir / "metrics" / "step6b4_full_alignment_qc.json").as_posix())
    set_prop(status, "step6b4_full_alignment_summary_json", (run_dir / "metrics" / "step6b4_full_alignment_summary.json").as_posix())
    set_prop(status, "step6b4_loss_compatibility_smoke_summary_json", (run_dir / "metrics" / "step6b4_loss_compatibility_smoke_summary.json").as_posix())
    set_prop(status, "step6b4_main_ready_config_stub", main_config.as_posix())
    set_prop(status, "step6b4_control_ready_config_stub", control_config.as_posix())
    set_prop(status, "step6b4_expected_samples", alignment_summary.get("expected_samples"))
    set_prop(status, "step6b4_aligned_samples", alignment_summary.get("aligned_samples"))
    set_prop(status, "step6b4_failed_outputs", alignment_summary.get("failed_outputs"))
    set_prop(status, "step6b4_min_finite_ratio", alignment_summary.get("finite_ratio_min"))
    set_prop(status, "step6b4_loss_smoke_status", loss_summary.get("status"))
    set_prop(status, "generated_at", now_utc())
    write_json(status_path, status)


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 6B4 multi-sample real topography loss smoke.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--max-samples", type=int, default=12)
    args = parser.parse_args()

    try:
        rows = read_csv(args.manifest)
        selected = select_smoke_rows(rows, args.seed, args.max_samples)
        if not selected:
            raise ValueError("No valid aligned rows available for smoke test.")

        results = []
        for index, row in enumerate(selected):
            result = run_one(row, args.seed + index)
            results.append(result)
            print(f"[{index + 1}/{len(selected)}] {row['tile_id']} {result['status']}", flush=True)

        passed = [row for row in results if row["status"] == "passed"]
        loss_summary = {
            "step": "6B4",
            "status": "passed" if len(passed) == len(results) else "failed",
            "generated_at": now_utc(),
            "manifest": args.manifest.as_posix(),
            "tested_samples": len(results),
            "passed_samples": len(passed),
            "failed_samples": len(results) - len(passed),
            "split_counts": dict(Counter(row["split"] for row in results)),
            "event_location_counts": dict(Counter(row["event_location"] for row in results)),
            "loss_smoke_passed": len(passed) == len(results),
            "uses_real_topography": True,
            "uses_synthetic_logits": True,
            "training_started": False,
            "samples": results,
        }
        write_json(args.summary_json, loss_summary)

        alignment_summary = read_json_if_exists(args.run_dir / "metrics" / "step6b4_full_alignment_summary.json")
        main_config, control_config = write_config_stubs(REPO_ROOT, args.manifest)
        report_path, run_report_path = write_report(
            repo_root=REPO_ROOT,
            run_dir=args.run_dir,
            alignment_summary=alignment_summary,
            loss_summary=loss_summary,
            manifest_path=args.manifest,
            main_config=main_config,
            control_config=control_config,
        )
        update_pipeline_status(
            repo_root=REPO_ROOT,
            run_dir=args.run_dir,
            alignment_summary=alignment_summary,
            loss_summary=loss_summary,
            report_path=report_path,
            run_report_path=run_report_path,
            main_config=main_config,
            control_config=control_config,
        )

        print(f"status={loss_summary['status']}")
        print(f"tested_samples={loss_summary['tested_samples']}")
        print(f"summary={args.summary_json}")
        print(f"report={report_path}")
        return 0 if loss_summary["loss_smoke_passed"] else 2
    except Exception as exc:  # noqa: BLE001
        failure = {
            "step": "6B4",
            "status": "failed_exception",
            "generated_at": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "loss_smoke_passed": False,
            "training_started": False,
        }
        write_json(args.summary_json, failure)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
