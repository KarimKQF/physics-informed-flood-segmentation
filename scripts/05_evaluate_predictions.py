from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metrics.confusion import compute_confusion_matrix  # noqa: E402
from metrics.segmentation_metrics import compute_all_metrics  # noqa: E402

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - exercised only on missing dependency
    raise SystemExit(
        "rasterio is required to read GeoTIFF masks. Install with: "
        "python -m pip install rasterio"
    ) from exc


FULLY_INVALID_LABELHAND_TILE_IDS = {
    "Ghana_234935",
    "Ghana_26376",
    "Ghana_277",
    "Ghana_5079",
    "Ghana_83483",
}
SUPPORTED_PREDICTION_SUFFIXES = {".tif", ".tiff", ".npy", ".npz"}
METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "iou_background",
    "iou_water",
    "mean_iou",
    "support_background",
    "support_water",
    "valid_pixel_count",
    "tn",
    "fp",
    "fn",
    "tp",
]


def split_values(split_membership: str) -> list[str]:
    return [value for value in split_membership.split(";") if value]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, list):
        return json.dumps(value)
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_prediction_index(prediction_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in prediction_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_PREDICTION_SUFFIXES:
            index[path.stem.lower()].append(path)
    return {stem: sorted(paths) for stem, paths in index.items()}


def find_prediction_path(
    prediction_index: dict[str, list[Path]],
    tile_id: str,
) -> Path | None:
    normalized = tile_id.lower()
    exact_stems = [
        normalized,
        f"{normalized}_pred",
        f"{normalized}_prediction",
        f"{normalized}_mask",
    ]
    for stem in exact_stems:
        paths = prediction_index.get(stem)
        if paths:
            return sorted(paths, key=lambda path: (len(path.name), str(path)))[0]

    fallback: list[Path] = []
    for stem, paths in prediction_index.items():
        if stem.startswith(normalized) and "labelhand" not in stem:
            fallback.extend(paths)
    if not fallback:
        return None
    return sorted(fallback, key=lambda path: (len(path.stem), str(path)))[0]


def read_mask(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with rasterio.open(path) as dataset:
            return dataset.read(1)
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as data:
            if not data.files:
                raise ValueError(f"{path} contains no arrays.")
            return data[data.files[0]]
    raise ValueError(f"Unsupported prediction format: {path.suffix}")


def prepare_prediction(prediction: np.ndarray, *, threshold: float | None) -> np.ndarray:
    if threshold is not None:
        return (prediction >= threshold).astype(np.int64)
    return prediction


def row_matches_filters(row: dict[str, str], args: argparse.Namespace) -> bool:
    tile_id = row["tile_id"]
    if tile_id in FULLY_INVALID_LABELHAND_TILE_IDS:
        return False

    recommendation = row.get("cleaning_recommendation", "")
    if not args.include_exclude_candidates and recommendation == "exclude_candidate":
        return False
    if not args.include_exclude_candidates and recommendation not in {"keep", "warning_review"}:
        return False

    if args.split != "all" and args.split not in split_values(row.get("split_membership", "")):
        return False

    if args.event_location and row.get("event_location") not in args.event_location:
        return False

    return True


def metrics_from_matrix(matrix: np.ndarray, zero_division: str) -> dict[str, Any]:
    metrics = compute_all_metrics(confusion_matrix=matrix, zero_division=zero_division)
    tn = int(matrix[0, 0])
    fp = int(matrix[0, 1])
    fn = int(matrix[1, 0])
    tp = int(matrix[1, 1])
    metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return metrics


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key, "") for key in METRIC_COLUMNS}


def evaluate_tile(
    row: dict[str, str],
    prediction_path: Path,
    *,
    threshold: float | None,
    zero_division: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    tile_id = row["tile_id"]
    label_path = Path(row["hand_labeled_mask_path"])
    try:
        y_true = read_mask(label_path)
        y_pred = prepare_prediction(read_mask(prediction_path), threshold=threshold)
        matrix = compute_confusion_matrix(y_true, y_pred, num_classes=2, ignore_index=-1)
        metrics = metrics_from_matrix(matrix, zero_division=zero_division)
        result = {
            "tile_id": tile_id,
            "split_membership": row.get("split_membership", ""),
            "event_location": row.get("event_location", ""),
            "cleaning_recommendation": row.get("cleaning_recommendation", ""),
            "prediction_path": str(prediction_path).replace("\\", "/"),
            "label_path": str(label_path).replace("\\", "/"),
            "status": "ok",
            "error": "",
            **flatten_metrics(metrics),
        }
        return result, matrix
    except Exception as exc:  # noqa: BLE001 - CLI should continue and report per-tile errors.
        return (
            {
                "tile_id": tile_id,
                "split_membership": row.get("split_membership", ""),
                "event_location": row.get("event_location", ""),
                "cleaning_recommendation": row.get("cleaning_recommendation", ""),
                "prediction_path": str(prediction_path).replace("\\", "/"),
                "label_path": str(label_path).replace("\\", "/"),
                "status": "error",
                "error": str(exc),
            },
            None,
        )


def aggregate_group(
    group_type: str,
    group_value: str,
    results: list[dict[str, Any]],
    matrices: list[np.ndarray],
    *,
    zero_division: str,
) -> dict[str, Any]:
    ok_count = len(matrices)
    if matrices:
        matrix = np.stack(matrices, axis=0).sum(axis=0)
    else:
        matrix = np.zeros((2, 2), dtype=np.int64)
    metrics = metrics_from_matrix(matrix, zero_division=zero_division)
    return {
        "group_type": group_type,
        "group_value": group_value,
        "tile_count": len(results),
        "valid_tile_count": ok_count,
        "missing_prediction_count": sum(1 for result in results if result["status"] == "missing_prediction"),
        "error_count": sum(1 for result in results if result["status"] == "error"),
        **flatten_metrics(metrics),
    }


def build_grouped_metrics(
    results: list[dict[str, Any]],
    matrix_by_tile: dict[str, np.ndarray],
    *,
    zero_division: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[("global", "all")].append(result)
        splits = split_values(result.get("split_membership", "")) or ["unspecified"]
        for split in splits:
            groups[("split", split)].append(result)
        groups[("event_location", result.get("event_location", "unspecified") or "unspecified")].append(result)

    grouped_rows = []
    for (group_type, group_value), group_results in sorted(groups.items()):
        matrices = [
            matrix_by_tile[result["tile_id"]]
            for result in group_results
            if result["tile_id"] in matrix_by_tile
        ]
        grouped_rows.append(
            aggregate_group(
                group_type,
                group_value,
                group_results,
                matrices,
                zero_division=zero_division,
            )
        )
    return grouped_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary flood segmentation prediction masks against LabelHand masks."
    )
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest-csv",
        default=REPO_ROOT / "reports" / "sen1floods11_handlabeled_index.csv",
        type=Path,
    )
    parser.add_argument(
        "--audit-csv",
        default=REPO_ROOT / "reports" / "sen1floods11_handlabeled_audit.csv",
        type=Path,
        help="Optional STEP 2 audit CSV path recorded in the JSON summary.",
    )
    parser.add_argument("--output-csv", required=True, type=Path, help="Per-tile metrics CSV.")
    parser.add_argument(
        "--output-grouped-csv",
        required=True,
        type=Path,
        help="Grouped global/split/event metrics CSV.",
    )
    parser.add_argument("--output-summary-json", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test", "bolivia", "all"],
        default="all",
    )
    parser.add_argument(
        "--event-location",
        action="append",
        help="Optional event/location filter. Repeat for multiple values.",
    )
    parser.add_argument(
        "--include-exclude-candidates",
        action="store_true",
        help=(
            "Include manifest rows marked exclude_candidate. The five fully invalid "
            "LabelHand tiles remain excluded."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold for probability/logit prediction arrays.",
    )
    parser.add_argument("--zero-division", choices=["nan", "0", "1"], default="nan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv_rows(args.manifest_csv)
    filtered_rows = [row for row in rows if row_matches_filters(row, args)]
    prediction_index = build_prediction_index(args.prediction_dir)

    results: list[dict[str, Any]] = []
    matrix_by_tile: dict[str, np.ndarray] = {}
    for row in filtered_rows:
        tile_id = row["tile_id"]
        prediction_path = find_prediction_path(prediction_index, tile_id)
        if prediction_path is None:
            results.append(
                {
                    "tile_id": tile_id,
                    "split_membership": row.get("split_membership", ""),
                    "event_location": row.get("event_location", ""),
                    "cleaning_recommendation": row.get("cleaning_recommendation", ""),
                    "prediction_path": "",
                    "label_path": row.get("hand_labeled_mask_path", ""),
                    "status": "missing_prediction",
                    "error": "No prediction file matched this tile_id.",
                }
            )
            continue
        result, matrix = evaluate_tile(
            row,
            prediction_path,
            threshold=args.threshold,
            zero_division=args.zero_division,
        )
        results.append(result)
        if matrix is not None:
            matrix_by_tile[tile_id] = matrix

    grouped_rows = build_grouped_metrics(
        results,
        matrix_by_tile,
        zero_division=args.zero_division,
    )

    per_tile_fields = [
        "tile_id",
        "split_membership",
        "event_location",
        "cleaning_recommendation",
        "prediction_path",
        "label_path",
        "status",
        "error",
        *METRIC_COLUMNS,
    ]
    grouped_fields = [
        "group_type",
        "group_value",
        "tile_count",
        "valid_tile_count",
        "missing_prediction_count",
        "error_count",
        *METRIC_COLUMNS,
    ]
    write_csv_rows(args.output_csv, results, per_tile_fields)
    write_csv_rows(args.output_grouped_csv, grouped_rows, grouped_fields)

    summary = {
        "prediction_dir": str(args.prediction_dir).replace("\\", "/"),
        "manifest_csv": str(args.manifest_csv).replace("\\", "/"),
        "audit_csv": str(args.audit_csv).replace("\\", "/"),
        "split_filter": args.split,
        "event_location_filter": args.event_location or [],
        "include_exclude_candidates": args.include_exclude_candidates,
        "fully_invalid_labelhand_tiles_excluded": sorted(FULLY_INVALID_LABELHAND_TILE_IDS),
        "threshold": args.threshold,
        "zero_division": args.zero_division,
        "manifest_rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "evaluated_tiles": len(matrix_by_tile),
        "missing_prediction_count": sum(1 for result in results if result["status"] == "missing_prediction"),
        "error_count": sum(1 for result in results if result["status"] == "error"),
        "per_tile_metrics_csv": str(args.output_csv).replace("\\", "/"),
        "grouped_metrics_csv": str(args.output_grouped_csv).replace("\\", "/"),
        "grouped_metrics": grouped_rows,
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
