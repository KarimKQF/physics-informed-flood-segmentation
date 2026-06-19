"""STEP 2: index and audit the downloaded Sen1Floods11 dataset.

This script reads raw files in-place and writes reports/CSVs only. It does
not clean, delete, move, overwrite, or otherwise modify raw data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise SystemExit(
        "BLOCKED: rasterio is required for STEP 2 GeoTIFF audit.\n"
        "Install it with: python -m pip install rasterio\n"
        f"Original error: {exc}"
    ) from exc


HAND_LAYERS = {
    "S1Hand": "_S1Hand",
    "S2Hand": "_S2Hand",
    "LabelHand": "_LabelHand",
    "S1OtsuLabelHand": "_S1OtsuLabelHand",
    "JRCWaterHand": "_JRCWaterHand",
}
EXPECTED_HAND_LAYERS = ["S1Hand", "S2Hand", "LabelHand"]
EXPECTED_MASK_VALUES = {-1, 0, 1}
HIGH_INVALID_RATIO = 0.5
GEO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RasterAudit:
    readable: bool
    error: str
    width: int | None
    height: int | None
    bands: int | None
    dtype: str
    crs: str
    transform: str
    resolution: str
    nodata: str
    min_values: str
    max_values: str
    nan_count: int | None
    nan_ratio: float | None
    inf_count: int | None
    inf_ratio: float | None
    unique_values: str
    label_counts_json: str
    invalid_pct: float | None
    non_water_pct: float | None
    water_pct: float | None
    unexpected_values: str
    empty_mask: bool | None
    only_invalid: bool | None
    no_water: bool | None


def bytes_to_gb(value: int | float) -> float:
    return round(float(value) / 1_000_000_000, 3)


def pct(part: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return round(float(part) * 100.0 / float(total), 6)


def fmt_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def fmt_float_tuple(values: Any) -> str:
    return ",".join(f"{float(value):.12g}" for value in values)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    if not value:
        return ()
    try:
        return tuple(float(part) for part in value.split(","))
    except ValueError:
        return ()


def allclose_tuple(left: str, right: str, tolerance: float = GEO_TOLERANCE) -> bool:
    left_values = parse_float_tuple(left)
    right_values = parse_float_tuple(right)
    if len(left_values) != len(right_values):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(left_values, right_values, strict=True))


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config: {path}")
    return payload


def classify_file(relative: Path) -> tuple[str, dict[str, bool]]:
    parts = set(relative.parts)
    text = relative.as_posix()
    flags = {
        "is_hand_labeled": "HandLabeled" in parts,
        "is_weak_labeled": "WeaklyLabeled" in parts,
        "is_perm_water": "perm_water" in parts,
        "is_splits": "splits" in parts,
        "is_catalog": "catalog" in parts or relative.name == "catalog.zip",
        "is_metadata": relative.name == "Sen1Floods11_Metadata.geojson",
        "is_checkpoints": "checkpoints" in parts,
    }
    if flags["is_splits"]:
        category = "splits"
    elif flags["is_hand_labeled"]:
        category = "HandLabeled"
    elif flags["is_weak_labeled"]:
        category = "WeaklyLabeled"
    elif flags["is_perm_water"]:
        category = "perm_water"
    elif flags["is_catalog"]:
        category = "catalog"
    elif flags["is_metadata"]:
        category = "metadata"
    elif flags["is_checkpoints"]:
        category = "checkpoints"
    else:
        category = "other"
    if "v1.1/data/flood_events" in text and category == "other":
        category = "flood_events_other"
    return category, flags


def infer_layer(path: Path, relative: Path) -> str:
    if path.parent.name:
        return path.parent.name
    return ""


def infer_tile_id(filename: str, layer: str) -> str:
    stem = Path(filename).stem
    known_suffixes = sorted(HAND_LAYERS.values(), key=len, reverse=True)
    extra_suffixes = [
        "_S1Weak",
        "_S2Weak",
        "_S1OtsuLabelWeak",
        "_S2IndexLabelWeak",
        "_label",
    ]
    for suffix in [*known_suffixes, *extra_suffixes]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    if layer and stem.endswith(f"_{layer}"):
        return stem[: -(len(layer) + 1)]
    return stem


def event_from_tile(tile_id: str) -> str:
    if "_" in tile_id:
        return tile_id.split("_", maxsplit=1)[0]
    return tile_id


def build_inventory(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(dataset_root)
        category, flags = classify_file(relative)
        layer = infer_layer(path, relative)
        tile_id = infer_tile_id(path.name, layer)
        row = {
            "file_path": fmt_path(path),
            "relative_path": relative.as_posix(),
            "subset_category": category,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "file_size_mb": round(path.stat().st_size / 1_000_000, 6),
            "inferred_event_or_location": event_from_tile(tile_id),
            "inferred_tile_id": tile_id,
            "inferred_modality_or_layer": layer,
        }
        row.update(flags)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_to_external(local_path: Path, external_path: Path) -> None:
    external_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, external_path)


def collect_hand_files(hand_root: Path) -> dict[str, dict[str, Path]]:
    samples: dict[str, dict[str, Path]] = defaultdict(dict)
    for layer, suffix in HAND_LAYERS.items():
        layer_root = hand_root / layer
        if not layer_root.exists():
            continue
        for path in sorted(layer_root.glob("*.tif")):
            tile_id = infer_tile_id(path.name, layer)
            samples[tile_id][layer] = path
    return dict(samples)


def parse_split_csv(path: Path, split_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            image_name = row[0].strip()
            mask_name = row[1].strip() if len(row) > 1 else ""
            tile_id = infer_tile_id(image_name, "S1Hand")
            rows.append(
                {
                    "split": split_name,
                    "split_file": fmt_path(path),
                    "image_name": image_name,
                    "mask_name": mask_name,
                    "tile_id": tile_id,
                }
            )
    return rows


def load_hand_splits(splits_root: Path) -> tuple[list[dict[str, str]], dict[str, set[str]], dict[str, int]]:
    split_files = {
        "train": splits_root / "flood_train_data.csv",
        "valid": splits_root / "flood_valid_data.csv",
        "test": splits_root / "flood_test_data.csv",
        "bolivia": splits_root / "flood_bolivia_data.csv",
    }
    records: list[dict[str, str]] = []
    by_tile: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = {}
    for split_name, path in split_files.items():
        if not path.exists():
            counts[split_name] = 0
            continue
        split_records = parse_split_csv(path, split_name)
        counts[split_name] = len(split_records)
        records.extend(split_records)
        for record in split_records:
            by_tile[record["tile_id"]].add(split_name)
    return records, dict(by_tile), counts


def _json_counter(counter: Counter[Any]) -> str:
    return json.dumps({str(key): int(value) for key, value in sorted(counter.items())}, sort_keys=True)


def audit_raster(path: Path, *, is_mask: bool) -> RasterAudit:
    try:
        with rasterio.open(path) as dataset:
            width = dataset.width
            height = dataset.height
            bands = dataset.count
            dtype = ",".join(dataset.dtypes)
            crs = str(dataset.crs) if dataset.crs else ""
            transform = fmt_float_tuple(tuple(dataset.transform))
            resolution = fmt_float_tuple(dataset.res) if dataset.res else ""
            nodata = str(dataset.nodata) if dataset.nodata is not None else ""
            mins: list[str] = []
            maxs: list[str] = []
            unique_values: Counter[Any] = Counter()
            total_pixels = 0
            nan_count = 0
            inf_count = 0

            for band_index in range(1, bands + 1):
                data = dataset.read(band_index, masked=False)
                total_pixels += int(data.size)
                if np.issubdtype(data.dtype, np.floating):
                    nan_count += int(np.isnan(data).sum())
                    inf_count += int(np.isinf(data).sum())
                    valid = data[np.isfinite(data)]
                else:
                    valid = data
                if valid.size == 0:
                    mins.append("")
                    maxs.append("")
                else:
                    mins.append(str(float(np.min(valid))))
                    maxs.append(str(float(np.max(valid))))
                if is_mask:
                    values, counts = np.unique(data, return_counts=True)
                    for value, count in zip(values.tolist(), counts.tolist(), strict=True):
                        if isinstance(value, np.generic):
                            value = value.item()
                        unique_values[value] += int(count)

            if is_mask:
                mask_pixels = sum(unique_values.values())
                invalid_count = unique_values.get(-1, 0)
                non_water_count = unique_values.get(0, 0)
                water_count = unique_values.get(1, 0)
                unexpected = sorted(
                    str(key) for key in unique_values if key not in EXPECTED_MASK_VALUES
                )
                only_invalid = mask_pixels > 0 and invalid_count == mask_pixels
                no_water = water_count == 0
                empty_mask = mask_pixels == 0
                unique_text = ",".join(str(key) for key in sorted(unique_values, key=str))
                label_counts_json = _json_counter(unique_values)
                invalid_pct = pct(invalid_count, mask_pixels)
                non_water_pct = pct(non_water_count, mask_pixels)
                water_pct = pct(water_count, mask_pixels)
            else:
                unique_text = ""
                label_counts_json = ""
                invalid_pct = None
                non_water_pct = None
                water_pct = None
                unexpected = []
                empty_mask = None
                only_invalid = None
                no_water = None

            return RasterAudit(
                readable=True,
                error="",
                width=width,
                height=height,
                bands=bands,
                dtype=dtype,
                crs=crs,
                transform=transform,
                resolution=resolution,
                nodata=nodata,
                min_values=";".join(mins),
                max_values=";".join(maxs),
                nan_count=nan_count,
                nan_ratio=round(nan_count / total_pixels, 8) if total_pixels else 0.0,
                inf_count=inf_count,
                inf_ratio=round(inf_count / total_pixels, 8) if total_pixels else 0.0,
                unique_values=unique_text,
                label_counts_json=label_counts_json,
                invalid_pct=invalid_pct,
                non_water_pct=non_water_pct,
                water_pct=water_pct,
                unexpected_values=",".join(unexpected),
                empty_mask=empty_mask,
                only_invalid=only_invalid,
                no_water=no_water,
            )
    except Exception as exc:  # noqa: BLE001
        return RasterAudit(
            readable=False,
            error=repr(exc),
            width=None,
            height=None,
            bands=None,
            dtype="",
            crs="",
            transform="",
            resolution="",
            nodata="",
            min_values="",
            max_values="",
            nan_count=None,
            nan_ratio=None,
            inf_count=None,
            inf_ratio=None,
            unique_values="",
            label_counts_json="",
            invalid_pct=None,
            non_water_pct=None,
            water_pct=None,
            unexpected_values="",
            empty_mask=None,
            only_invalid=None,
            no_water=None,
        )


def compare_rasters(reference: RasterAudit, other: RasterAudit) -> dict[str, bool]:
    if not reference.readable or not other.readable:
        return {
            "dimension_match": False,
            "crs_match": False,
            "transform_match": False,
            "resolution_match": False,
        }
    return {
        "dimension_match": reference.width == other.width and reference.height == other.height,
        "crs_match": reference.crs == other.crs,
        "transform_match": allclose_tuple(reference.transform, other.transform),
        "resolution_match": allclose_tuple(reference.resolution, other.resolution),
    }


def build_hand_index_and_audit(
    samples: dict[str, dict[str, Path]],
    split_by_tile: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    aggregate_label_counts: Counter[Any] = Counter()
    anomaly_counter: Counter[str] = Counter()
    recommendation_counter: Counter[str] = Counter()

    for tile_id in sorted(samples):
        layers = samples[tile_id]
        missing = [layer for layer in EXPECTED_HAND_LAYERS if layer not in layers]
        splits = sorted(split_by_tile.get(tile_id, set()))
        anomalies: list[str] = []
        if "S1Hand" not in layers:
            anomalies.append("missing_s1")
        if "S2Hand" not in layers:
            anomalies.append("missing_s2")
        if "LabelHand" not in layers:
            anomalies.append("missing_mask")
        if not splits:
            anomalies.append("not_in_split")

        audits: dict[str, RasterAudit] = {}
        for layer, path in sorted(layers.items()):
            is_mask = layer in {"LabelHand", "S1OtsuLabelHand", "JRCWaterHand"}
            raster_audit = audit_raster(path, is_mask=is_mask)
            audits[layer] = raster_audit
            if not raster_audit.readable:
                anomalies.append(f"unreadable_{layer}")
            if is_mask and raster_audit.readable:
                if raster_audit.label_counts_json:
                    aggregate_label_counts.update(
                        {int(k): int(v) for k, v in json.loads(raster_audit.label_counts_json).items()}
                    )
                if raster_audit.unexpected_values:
                    anomalies.append(f"unexpected_values_{layer}")
                if raster_audit.empty_mask:
                    anomalies.append(f"empty_mask_{layer}")
                if raster_audit.only_invalid:
                    anomalies.append(f"only_invalid_{layer}")
                if raster_audit.no_water and layer == "LabelHand":
                    anomalies.append("no_water")
                if (
                    raster_audit.invalid_pct is not None
                    and raster_audit.invalid_pct > HIGH_INVALID_RATIO * 100
                    and layer == "LabelHand"
                ):
                    anomalies.append("high_invalid_ratio")

            audit_rows.append(
                {
                    "tile_id": tile_id,
                    "event_location": event_from_tile(tile_id),
                    "layer": layer,
                    "file_path": fmt_path(path),
                    **raster_audit.__dict__,
                }
            )

        if "S1Hand" in audits and "LabelHand" in audits:
            matches = compare_rasters(audits["S1Hand"], audits["LabelHand"])
            if not matches["dimension_match"]:
                anomalies.append("dimension_mismatch_s1_mask")
            if not matches["crs_match"]:
                anomalies.append("crs_mismatch_s1_mask")
            if not matches["transform_match"] or not matches["resolution_match"]:
                anomalies.append("transform_or_resolution_mismatch_s1_mask")
        if "S2Hand" in audits and "LabelHand" in audits:
            matches = compare_rasters(audits["S2Hand"], audits["LabelHand"])
            if not matches["dimension_match"]:
                anomalies.append("dimension_mismatch_s2_mask")
            if not matches["crs_match"]:
                anomalies.append("crs_mismatch_s2_mask")
            if not matches["transform_match"] or not matches["resolution_match"]:
                anomalies.append("transform_or_resolution_mismatch_s2_mask")

        unique_anomalies = sorted(set(anomalies))
        for anomaly in unique_anomalies:
            anomaly_counter[anomaly] += 1

        if any(anomaly.startswith("unreadable") for anomaly in unique_anomalies):
            recommendation = "corrupted_or_unreadable"
            status = "error"
        elif any(
            anomaly in unique_anomalies
            for anomaly in [
                "missing_s1",
                "missing_mask",
                "dimension_mismatch_s1_mask",
                "unexpected_values_LabelHand",
                "empty_mask_LabelHand",
                "only_invalid_LabelHand",
            ]
        ):
            recommendation = "exclude_candidate"
            status = "error"
        elif unique_anomalies:
            recommendation = "warning_review"
            status = "warning"
        else:
            recommendation = "keep"
            status = "ok"
        recommendation_counter[recommendation] += 1

        index_rows.append(
            {
                "tile_id": tile_id,
                "event_location": event_from_tile(tile_id),
                "available_modalities_layers": ";".join(sorted(layers)),
                "sentinel1_path": fmt_path(layers.get("S1Hand", "")),
                "sentinel2_path": fmt_path(layers.get("S2Hand", "")),
                "hand_labeled_mask_path": fmt_path(layers.get("LabelHand", "")),
                "s1_otsu_qc_path": fmt_path(layers.get("S1OtsuLabelHand", "")),
                "jrc_water_qc_path": fmt_path(layers.get("JRCWaterHand", "")),
                "split_membership": ";".join(splits),
                "missing_files": ";".join(missing),
                "anomalies": ";".join(unique_anomalies),
                "cleaning_recommendation": recommendation,
                "status": status,
            }
        )

    summary = {
        "aggregate_label_counts": aggregate_label_counts,
        "anomaly_counter": anomaly_counter,
        "recommendation_counter": recommendation_counter,
    }
    return index_rows, audit_rows, summary


def summarize_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(row["subset_category"] for row in rows)
    category_sizes = Counter()
    extension_counts = Counter(row["extension"] for row in rows)
    for row in rows:
        category_sizes[row["subset_category"]] += int(row["file_size_bytes"])
    return {
        "total_files": len(rows),
        "total_size_bytes": sum(int(row["file_size_bytes"]) for row in rows),
        "category_counts": category_counts,
        "category_sizes": category_sizes,
        "extension_counts": extension_counts,
    }


def build_report(
    *,
    generated_at: str,
    status: str,
    dataset_root: Path,
    reports_dir: Path,
    local_files: dict[str, Path],
    external_files: dict[str, Path],
    inventory_summary: dict[str, Any],
    index_rows: list[dict[str, Any]],
    split_records: list[dict[str, str]],
    all_split_files: list[Path],
    split_counts: dict[str, int],
    split_not_matched: int,
    files_not_used_train_valid_test: int,
    files_not_used_any_split: int,
    audit_summary: dict[str, Any],
    blocking_reasons: list[str],
) -> str:
    index_status_counts = Counter(row["status"] for row in index_rows)
    label_counts: Counter[Any] = audit_summary["aggregate_label_counts"]
    total_label_pixels = sum(label_counts.values())

    lines = [
        "# STEP 2 - Sen1Floods11 indexing and audit report",
        "",
        "## Summary",
        f"- Status: `{status}`",
        f"- Generated at: `{generated_at}`",
        "- Raw data modified: `false`",
        "- STEP 3 started: `false`",
        "- Next step allowed: `false`",
        f"- Dataset root: `{fmt_path(dataset_root)}`",
        "",
        "## Downloaded dataset structure",
        "| Category | Files | Size GB |",
        "|---|---:|---:|",
    ]
    for category, count in inventory_summary["category_counts"].most_common():
        size = inventory_summary["category_sizes"][category]
        lines.append(f"| `{category}` | {count} | {bytes_to_gb(size)} |")

    lines.extend(
        [
            "",
            "## Inventory totals",
            f"- Total files: `{inventory_summary['total_files']}`",
            f"- Total size: `{bytes_to_gb(inventory_summary['total_size_bytes'])} GB`",
            f"- Extensions: `{dict(inventory_summary['extension_counts'])}`",
            "",
            "## Hand-labeled index totals",
            f"- Samples indexed: `{len(index_rows)}`",
            f"- Status counts: `{dict(index_status_counts)}`",
            f"- Cleaning recommendation counts: `{dict(audit_summary['recommendation_counter'])}`",
            "",
            "## Split totals",
            f"- Split files found: `{len(all_split_files)}`",
            f"- Train samples: `{split_counts.get('train', 0)}`",
            f"- Validation samples: `{split_counts.get('valid', 0)}`",
            f"- Test samples: `{split_counts.get('test', 0)}`",
            f"- Bolivia holdout samples: `{split_counts.get('bolivia', 0)}`",
            f"- Samples in split files not matched to indexed files: `{split_not_matched}`",
            f"- Hand-labeled samples not used by train/valid/test splits: `{files_not_used_train_valid_test}`",
            f"- Hand-labeled samples not used by any flood_handlabeled split: `{files_not_used_any_split}`",
            "- Split file names:",
            *[f"  - `{fmt_path(path)}`" for path in all_split_files],
            "",
            "## Mask label distribution",
            "| Label | Count | Percent | Interpretation |",
            "|---:|---:|---:|---|",
        ]
    )
    interpretations = {-1: "invalid/no-data", 0: "non-water", 1: "water"}
    for label in sorted(label_counts):
        lines.append(
            f"| `{label}` | {label_counts[label]} | {pct(label_counts[label], total_label_pixels)} | "
            f"{interpretations.get(int(label), 'unexpected')} |"
        )

    anomaly_counter: Counter[str] = audit_summary["anomaly_counter"]
    lines.extend(
        [
            "",
            "## Image/mask consistency findings",
            f"- Dimension mismatches S1/mask: `{anomaly_counter.get('dimension_mismatch_s1_mask', 0)}`",
            f"- Dimension mismatches S2/mask: `{anomaly_counter.get('dimension_mismatch_s2_mask', 0)}`",
            f"- CRS mismatches S1/mask: `{anomaly_counter.get('crs_mismatch_s1_mask', 0)}`",
            f"- CRS mismatches S2/mask: `{anomaly_counter.get('crs_mismatch_s2_mask', 0)}`",
            f"- Transform/resolution mismatches S1/mask: `{anomaly_counter.get('transform_or_resolution_mismatch_s1_mask', 0)}`",
            f"- Transform/resolution mismatches S2/mask: `{anomaly_counter.get('transform_or_resolution_mismatch_s2_mask', 0)}`",
            "",
            "## Anomaly summary",
            "| Anomaly | Samples |",
            "|---|---:|",
        ]
    )
    if anomaly_counter:
        for anomaly, count in anomaly_counter.most_common():
            lines.append(f"| `{anomaly}` | {count} |")
    else:
        lines.append("| `none` | 0 |")

    lines.extend(
        [
            "",
            "## Cleaning recommendations",
            "- `keep`: no anomaly detected by this audit.",
            "- `warning_review`: sample is readable but should be reviewed before training.",
            "- `exclude_candidate`: sample has missing critical files, label problems, or alignment problems.",
            "- `corrupted_or_unreadable`: at least one file could not be opened by rasterio.",
            "- No cleaning was applied in STEP 2.",
            "",
            "## Open questions",
            "- Confirm whether the Bolivia holdout should remain separate from train/validation/test in future experiments.",
            "- Confirm whether `S1OtsuLabelHand` and `JRCWaterHand` should be treated only as QC/reference layers or included in downstream audits.",
            "- Confirm the policy for no-water samples before STEP 3/cleaning decisions.",
            "",
            "## Generated files",
        ]
    )
    for label, path in local_files.items():
        lines.append(f"- Local {label}: `{fmt_path(path)}`")
    for label, path in external_files.items():
        lines.append(f"- External {label}: `{fmt_path(path)}`")

    lines.extend(["", "## Problems detected"])
    if blocking_reasons:
        lines.extend(f"- {reason}" for reason in blocking_reasons)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Decision required before STEP 3",
            "- Validate the audit findings and cleaning recommendation categories.",
            "- Do not start statistics/visualizations until this report is reviewed.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sen1floods11.yaml"))
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dataset_root = Path(str(config["raw"]["sen1floods11"]))  # type: ignore[index]
    reports_dir = Path(str(config["reports"]))
    external_project_root = Path(str(config["project_root"]))
    local_reports_dir = repo_root / "reports"

    generated_at = datetime.now().isoformat(timespec="seconds")
    blocking_reasons: list[str] = []
    expected_paths = [
        dataset_root / "v1.1" / "data" / "flood_events" / "HandLabeled",
        dataset_root / "v1.1" / "data" / "flood_events" / "WeaklyLabeled",
        dataset_root / "v1.1" / "data" / "perm_water",
        dataset_root / "v1.1" / "splits",
        dataset_root / "v1.1" / "catalog",
        dataset_root / "v1.1" / "Sen1Floods11_Metadata.geojson",
    ]
    for path in expected_paths:
        if not path.exists():
            blocking_reasons.append(f"Expected path missing: {fmt_path(path)}")
    all_split_files = sorted((dataset_root / "v1.1" / "splits").rglob("*.csv"))

    if blocking_reasons:
        status = "blocked"
        indexed = False
        audited = False
        inventory_rows: list[dict[str, Any]] = []
        index_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        split_records: list[dict[str, str]] = []
        all_split_files = []
        split_counts: dict[str, int] = {}
        audit_summary = {
            "aggregate_label_counts": Counter(),
            "anomaly_counter": Counter(),
            "recommendation_counter": Counter(),
        }
        split_not_matched = 0
        files_not_used_train_valid_test = 0
        files_not_used_any_split = 0
    else:
        inventory_rows = build_inventory(dataset_root)
        hand_root = dataset_root / "v1.1" / "data" / "flood_events" / "HandLabeled"
        samples = collect_hand_files(hand_root)
        split_records, split_by_tile, split_counts = load_hand_splits(
            dataset_root / "v1.1" / "splits" / "flood_handlabeled"
        )
        index_rows, audit_rows, audit_summary = build_hand_index_and_audit(samples, split_by_tile)
        indexed_tile_ids = {row["tile_id"] for row in index_rows}
        split_tile_ids = {record["tile_id"] for record in split_records}
        train_valid_test_tile_ids = {
            record["tile_id"]
            for record in split_records
            if record["split"] in {"train", "valid", "test"}
        }
        split_not_matched = len(split_tile_ids - indexed_tile_ids)
        files_not_used_train_valid_test = len(indexed_tile_ids - train_valid_test_tile_ids)
        files_not_used_any_split = len(indexed_tile_ids - split_tile_ids)
        status = "done"
        indexed = True
        audited = True

    inventory_summary = summarize_inventory(inventory_rows)

    local_files = {
        "inventory": local_reports_dir / "sen1floods11_file_inventory.csv",
        "hand index": local_reports_dir / "sen1floods11_handlabeled_index.csv",
        "hand audit": local_reports_dir / "sen1floods11_handlabeled_audit.csv",
        "report": local_reports_dir / "STEP_2_sen1floods11_audit_report.md",
    }
    external_files = {
        "inventory": reports_dir / "sen1floods11_file_inventory.csv",
        "hand index": reports_dir / "sen1floods11_handlabeled_index.csv",
        "hand audit": reports_dir / "sen1floods11_handlabeled_audit.csv",
        "report": reports_dir / "STEP_2_sen1floods11_audit_report.md",
    }

    local_reports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if inventory_rows:
        write_csv(local_files["inventory"], inventory_rows)
        copy_to_external(local_files["inventory"], external_files["inventory"])
    if index_rows:
        write_csv(local_files["hand index"], index_rows)
        copy_to_external(local_files["hand index"], external_files["hand index"])
    if audit_rows:
        write_csv(local_files["hand audit"], audit_rows)
        copy_to_external(local_files["hand audit"], external_files["hand audit"])

    report = build_report(
        generated_at=generated_at,
        status=status,
        dataset_root=dataset_root,
        reports_dir=reports_dir,
        local_files=local_files,
        external_files=external_files,
        inventory_summary=inventory_summary,
        index_rows=index_rows,
        split_records=split_records,
        all_split_files=all_split_files,
        split_counts=split_counts,
        split_not_matched=split_not_matched,
        files_not_used_train_valid_test=files_not_used_train_valid_test,
        files_not_used_any_split=files_not_used_any_split,
        audit_summary=audit_summary,
        blocking_reasons=blocking_reasons,
    )
    local_files["report"].write_text(report, encoding="utf-8")
    external_files["report"].write_text(report, encoding="utf-8")

    status_payload: dict[str, Any] = {
        "current_step": "2",
        "status": status,
        "indexed": indexed,
        "audited": audited,
        "raw_data_modified": False,
        "dataset_root": fmt_path(dataset_root),
        "inventory_csv": fmt_path(local_files["inventory"]),
        "handlabeled_index_csv": fmt_path(local_files["hand index"]),
        "handlabeled_audit_csv": fmt_path(local_files["hand audit"]),
        "audit_report": fmt_path(local_files["report"]),
        "inventory_total_files": inventory_summary["total_files"],
        "inventory_total_size_gb": bytes_to_gb(inventory_summary["total_size_bytes"]),
        "handlabeled_samples": len(index_rows),
        "split_counts": split_counts,
        "anomaly_counts": dict(audit_summary["anomaly_counter"]),
        "cleaning_recommendation_counts": dict(audit_summary["recommendation_counter"]),
        "blocking_reasons": blocking_reasons,
        "next_step_allowed": False,
        "human_validation_required": True,
        "generated_at": generated_at,
    }
    write_status(repo_root / "pipeline_status.json", status_payload)
    write_status(external_project_root / "pipeline_status.json", status_payload)

    print("\nSTEP 2 - Sen1Floods11 indexing and audit summary")
    print(f"Status: {status}")
    print(f"Dataset root: {fmt_path(dataset_root)}")
    print(f"Inventory files: {inventory_summary['total_files']}")
    print(f"Hand-labeled samples: {len(index_rows)}")
    print(f"Split counts: {split_counts}")
    print(f"Raw data modified: false")
    print(f"Report: {fmt_path(local_files['report'])}")
    print("STOP: Human validation required before STEP 3.")

    return 0 if status == "done" else 2


if __name__ == "__main__":
    raise SystemExit(main())
