"""STEP 3: Sen1Floods11 statistics and visualizations.

This script reads STEP 2 CSV outputs and raw raster files for qualitative
panels only. It writes statistics, figures, and reports. It does not modify,
move, clean, delete, or overwrite raw data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml


EXCLUDE_CANDIDATE_TILE_IDS = {
    "Ghana_234935",
    "Ghana_26376",
    "Ghana_277",
    "Ghana_5079",
    "Ghana_83483",
}
SPLIT_ORDER = ["train", "valid", "test", "bolivia"]
RANDOM_SEED = 42


def fmt_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def to_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: str | int | None, default: int = 0) -> int:
    if value in {None, ""}:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def copy_to_external(local_path: Path, external_path: Path) -> None:
    external_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, external_path)


def parse_semicolon_floats(value: str) -> list[float]:
    if not value:
        return []
    values: list[float] = []
    for part in value.split(";"):
        try:
            values.append(float(part))
        except ValueError:
            values.append(float("nan"))
    return values


def parse_label_counts(value: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not value:
        return counts
    payload = json.loads(value)
    for key, count in payload.items():
        counts[int(key)] += int(count)
    return counts


def split_values(split_membership: str) -> list[str]:
    return [item for item in split_membership.split(";") if item]


def first_split(split_membership: str) -> str:
    values = split_values(split_membership)
    return values[0] if values else "unsplit"


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    clean = np.array([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0:
        return {"min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    return {
        "min": round(float(np.min(clean)), 6),
        "p25": round(float(np.percentile(clean, 25)), 6),
        "median": round(float(np.percentile(clean, 50)), 6),
        "mean": round(float(np.mean(clean)), 6),
        "p75": round(float(np.percentile(clean, 75)), 6),
        "max": round(float(np.max(clean)), 6),
    }


def counts_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter(row.get(key, "") or "unknown" for row in rows))


def label_rows_by_tile(audit_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["tile_id"]: row for row in audit_rows if row["layer"] == "LabelHand"}


def audit_rows_by_tile_layer(audit_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["tile_id"], row["layer"]): row for row in audit_rows}


def selected_training_rows(index_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in index_rows
        if row["cleaning_recommendation"] in {"keep", "warning_review"}
    ]


def build_global_stats(
    index_rows: list[dict[str, str]],
    label_by_tile: dict[str, dict[str, str]],
) -> dict[str, Any]:
    samples_by_split: Counter[str] = Counter()
    for row in index_rows:
        for split in split_values(row["split_membership"]):
            samples_by_split[split] += 1

    anomaly_counts = Counter()
    for row in index_rows:
        anomalies = set(item for item in row["anomalies"].split(";") if item)
        if "no_water" in anomalies:
            anomaly_counts["no_water"] += 1
        if "high_invalid_ratio" in anomalies:
            anomaly_counts["high_invalid_ratio"] += 1
        if "only_invalid_LabelHand" in anomalies:
            anomaly_counts["only_invalid_LabelHand"] += 1

    training_candidates = selected_training_rows(index_rows)
    return {
        "total_hand_labeled_samples": len(index_rows),
        "analysis_training_candidate_samples": len(training_candidates),
        "exclude_candidate_samples": sorted(EXCLUDE_CANDIDATE_TILE_IDS),
        "samples_by_split": dict(samples_by_split),
        "samples_by_status": counts_by(index_rows, "status"),
        "samples_by_cleaning_recommendation": counts_by(index_rows, "cleaning_recommendation"),
        "samples_by_event_location": counts_by(index_rows, "event_location"),
        "no_water_samples": anomaly_counts["no_water"],
        "high_invalid_ratio_samples": anomaly_counts["high_invalid_ratio"],
        "only_invalid_labelhand_samples": anomaly_counts["only_invalid_LabelHand"],
        "labelhand_rows_available": len(label_by_tile),
    }


def build_label_stats(
    index_rows: list[dict[str, str]],
    label_by_tile: dict[str, dict[str, str]],
) -> dict[str, Any]:
    total_counts: Counter[int] = Counter()
    by_split: dict[str, Counter[int]] = defaultdict(Counter)
    by_event: dict[str, Counter[int]] = defaultdict(Counter)
    water_pct_by_tile: list[float] = []
    invalid_pct_by_tile: list[float] = []
    per_tile: dict[str, dict[str, Any]] = {}

    for row in index_rows:
        label_row = label_by_tile.get(row["tile_id"])
        if not label_row:
            continue
        counts = parse_label_counts(label_row["label_counts_json"])
        total_counts.update(counts)
        split = first_split(row["split_membership"])
        event = row["event_location"]
        by_split[split].update(counts)
        by_event[event].update(counts)
        water_pct = to_float(label_row["water_pct"])
        invalid_pct = to_float(label_row["invalid_pct"])
        water_pct_by_tile.append(water_pct)
        invalid_pct_by_tile.append(invalid_pct)
        per_tile[row["tile_id"]] = {
            "split": split,
            "event_location": event,
            "water_pct": water_pct,
            "invalid_pct": invalid_pct,
            "recommendation": row["cleaning_recommendation"],
            "status": row["status"],
        }

    total_pixels = sum(total_counts.values())

    def distribution(counter: Counter[int]) -> dict[str, dict[str, float | int]]:
        total = sum(counter.values())
        return {
            str(label): {
                "count": int(counter[label]),
                "percent": round(float(counter[label]) * 100.0 / total, 6) if total else 0.0,
            }
            for label in sorted(counter)
        }

    return {
        "total_label_counts": {str(key): int(value) for key, value in sorted(total_counts.items())},
        "total_label_distribution": distribution(total_counts),
        "total_pixels": int(total_pixels),
        "water_pct_by_tile_summary": percentile_summary(water_pct_by_tile),
        "invalid_pct_by_tile_summary": percentile_summary(invalid_pct_by_tile),
        "distribution_by_split": {
            split: distribution(counter) for split, counter in sorted(by_split.items())
        },
        "distribution_by_event_location": {
            event: distribution(counter) for event, counter in sorted(by_event.items())
        },
        "per_tile": per_tile,
    }


def build_image_stats(audit_rows: list[dict[str, str]], layer: str) -> dict[str, Any]:
    rows = [row for row in audit_rows if row["layer"] == layer]
    band_count = Counter(row["bands"] for row in rows)
    dtype_count = Counter(row["dtype"] for row in rows)
    nan_ratios = [to_float(row["nan_ratio"]) for row in rows]
    inf_ratios = [to_float(row["inf_ratio"]) for row in rows]
    band_mins: dict[int, list[float]] = defaultdict(list)
    band_maxs: dict[int, list[float]] = defaultdict(list)
    extreme_min = {"tile_id": None, "band": None, "value": None}
    extreme_max = {"tile_id": None, "band": None, "value": None}

    for row in rows:
        mins = parse_semicolon_floats(row["min_values"])
        maxs = parse_semicolon_floats(row["max_values"])
        for index, value in enumerate(mins, start=1):
            if np.isfinite(value):
                band_mins[index].append(value)
                if extreme_min["value"] is None or value < float(extreme_min["value"]):
                    extreme_min = {"tile_id": row["tile_id"], "band": index, "value": value}
        for index, value in enumerate(maxs, start=1):
            if np.isfinite(value):
                band_maxs[index].append(value)
                if extreme_max["value"] is None or value > float(extreme_max["value"]):
                    extreme_max = {"tile_id": row["tile_id"], "band": index, "value": value}

    return {
        "layer": layer,
        "sample_count": len(rows),
        "band_count_distribution": dict(band_count),
        "dtype_distribution": dict(dtype_count),
        "nan_ratio_summary": percentile_summary(nan_ratios),
        "inf_ratio_summary": percentile_summary(inf_ratios),
        "band_min_summary": {
            str(index): percentile_summary(values) for index, values in sorted(band_mins.items())
        },
        "band_max_summary": {
            str(index): percentile_summary(values) for index, values in sorted(band_maxs.items())
        },
        "extreme_min": extreme_min,
        "extreme_max": extreme_max,
    }


def save_current_figure(path: Path, external_figures_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    copy_to_external(path, external_figures_dir / path.name)


def bar_plot(counts: dict[str, int], title: str, ylabel: str, path: Path, external: Path) -> None:
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    width = max(7, min(18, len(labels) * 0.75))
    plt.figure(figsize=(width, 4.5))
    plt.bar(labels, values, color="#4C78A8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    for index, value in enumerate(values):
        plt.text(index, value, str(value), ha="center", va="bottom", fontsize=8)
    save_current_figure(path, external)


def histogram(values: list[float], title: str, xlabel: str, path: Path, external: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.hist(values, bins=30, color="#59A14F", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Samples")
    save_current_figure(path, external)


def box_by_group(
    grouped: dict[str, list[float]],
    title: str,
    ylabel: str,
    path: Path,
    external: Path,
    *,
    max_groups: int | None = None,
) -> None:
    items = [(key, values) for key, values in grouped.items() if values]
    items.sort(key=lambda item: (-len(item[1]), item[0]))
    if max_groups is not None:
        items = items[:max_groups]
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    width = max(7, min(18, len(labels) * 0.8))
    plt.figure(figsize=(width, 5))
    plt.boxplot(values, tick_labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    save_current_figure(path, external)


def minmax_boxplot(
    audit_rows: list[dict[str, str]],
    layer: str,
    title: str,
    path: Path,
    external: Path,
) -> None:
    rows = [row for row in audit_rows if row["layer"] == layer]
    band_mins: dict[int, list[float]] = defaultdict(list)
    band_maxs: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        for index, value in enumerate(parse_semicolon_floats(row["min_values"]), start=1):
            if np.isfinite(value):
                band_mins[index].append(value)
        for index, value in enumerate(parse_semicolon_floats(row["max_values"]), start=1):
            if np.isfinite(value):
                band_maxs[index].append(value)

    labels: list[str] = []
    data: list[list[float]] = []
    for index in sorted(set(band_mins) | set(band_maxs)):
        if band_mins.get(index):
            labels.append(f"B{index} min")
            data.append(band_mins[index])
        if band_maxs.get(index):
            labels.append(f"B{index} max")
            data.append(band_maxs[index])

    width = max(8, min(20, len(labels) * 0.55))
    plt.figure(figsize=(width, 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel("Raster value")
    plt.xticks(rotation=60, ha="right")
    save_current_figure(path, external)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype("float32")
    output = np.zeros_like(image, dtype="float32")
    if image.ndim == 2:
        values = image[np.isfinite(image)]
        if values.size == 0:
            return output
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((image - lo) / (hi - lo), 0, 1)
    for channel in range(image.shape[2]):
        values = image[:, :, channel]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        output[:, :, channel] = np.clip((values - lo) / (hi - lo), 0, 1)
    return output


def read_s1_visual(path: str) -> np.ndarray | None:
    if not path:
        return None
    with rasterio.open(path) as dataset:
        band1 = dataset.read(1)
        if dataset.count >= 2:
            band2 = dataset.read(2)
            return normalize_image(np.stack([band1, band2, (band1 + band2) / 2.0], axis=-1))
        return normalize_image(band1)


def read_s2_rgb(path: str) -> np.ndarray | None:
    if not path:
        return None
    with rasterio.open(path) as dataset:
        if dataset.count >= 4:
            indices = [4, 3, 2]
        elif dataset.count >= 3:
            indices = [1, 2, 3]
        else:
            return normalize_image(dataset.read(1))
        arrays = [dataset.read(index) for index in indices]
        return normalize_image(np.stack(arrays, axis=-1))


def read_mask(path: str) -> np.ndarray | None:
    if not path:
        return None
    with rasterio.open(path) as dataset:
        return dataset.read(1)


def show_mask(ax: plt.Axes, mask: np.ndarray | None, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if mask is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        return
    display = np.zeros((*mask.shape, 3), dtype="float32")
    display[mask == -1] = [0.45, 0.45, 0.45]
    display[mask == 0] = [0.92, 0.92, 0.92]
    display[mask == 1] = [0.1, 0.35, 0.9]
    other = ~np.isin(mask, [-1, 0, 1])
    display[other] = [0.9, 0.1, 0.1]
    ax.imshow(display)


def make_overlay(s1: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
    if s1 is None or mask is None:
        return None
    if s1.ndim == 2:
        base = np.stack([s1, s1, s1], axis=-1)
    else:
        base = s1.copy()
    overlay = base.copy()
    water = mask == 1
    invalid = mask == -1
    overlay[water] = overlay[water] * 0.45 + np.array([0.05, 0.3, 1.0]) * 0.55
    overlay[invalid] = overlay[invalid] * 0.35 + np.array([0.7, 0.7, 0.7]) * 0.65
    return np.clip(overlay, 0, 1)


def qualitative_panel(
    row: dict[str, str],
    label_row: dict[str, str] | None,
    category: str,
    path: Path,
    external: Path,
) -> None:
    s1 = read_s1_visual(row["sentinel1_path"])
    s2 = read_s2_rgb(row["sentinel2_path"])
    label = read_mask(row["hand_labeled_mask_path"])
    otsu = read_mask(row["s1_otsu_qc_path"])
    jrc = read_mask(row["jrc_water_qc_path"])
    overlay = make_overlay(s1, label)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.7))
    fig.suptitle(
        (
            f"{category} | {row['tile_id']} | split={row['split_membership']} | "
            f"event={row['event_location']} | water={to_float(label_row.get('water_pct') if label_row else 0):.2f}% | "
            f"invalid={to_float(label_row.get('invalid_pct') if label_row else 0):.2f}% | "
            f"{row['cleaning_recommendation']}"
        ),
        fontsize=10,
    )

    for ax, image, title in [
        (axes[0], s1, "Sentinel-1"),
        (axes[1], s2, "Sentinel-2 RGB"),
        (axes[5], overlay, "Mask overlay"),
    ]:
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if image is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        else:
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    show_mask(axes[2], label, "LabelHand")
    show_mask(axes[3], otsu, "S1OtsuLabelHand")
    show_mask(axes[4], jrc, "JRCWaterHand")

    save_current_figure(path, external)


def choose_samples(
    index_rows: list[dict[str, str]],
    label_by_tile: dict[str, dict[str, str]],
) -> list[tuple[str, dict[str, str]]]:
    rng = random.Random(RANDOM_SEED)
    rows_by_tile = {row["tile_id"]: row for row in index_rows}
    selections: list[tuple[str, dict[str, str]]] = []

    def sample_category(name: str, candidates: list[dict[str, str]], count: int) -> None:
        candidates = sorted(candidates, key=lambda row: row["tile_id"])
        if len(candidates) > count:
            candidates = rng.sample(candidates, count)
        for row in candidates:
            selections.append((name, row))

    sample_category(
        "random_keep",
        [row for row in index_rows if row["cleaning_recommendation"] == "keep"],
        4,
    )
    sample_category(
        "random_warning_review",
        [row for row in index_rows if row["cleaning_recommendation"] == "warning_review"],
        4,
    )
    sample_category(
        "no_water",
        [row for row in index_rows if "no_water" in row["anomalies"].split(";")],
        4,
    )
    sample_category(
        "high_invalid_ratio",
        [row for row in index_rows if "high_invalid_ratio" in row["anomalies"].split(";")],
        4,
    )
    for tile_id in sorted(EXCLUDE_CANDIDATE_TILE_IDS):
        row = rows_by_tile.get(tile_id)
        if row:
            selections.append(("exclude_candidate", row))
    for split in SPLIT_ORDER:
        candidates = [row for row in index_rows if split in split_values(row["split_membership"])]
        candidates.sort(key=lambda row: (row["cleaning_recommendation"] != "keep", row["tile_id"]))
        if candidates:
            selections.append((f"split_{split}", candidates[0]))

    unique: list[tuple[str, dict[str, str]]] = []
    seen_files: set[str] = set()
    for category, row in selections:
        filename = f"panel_{category}_{row['tile_id']}.png"
        if filename in seen_files:
            continue
        seen_files.add(filename)
        if row["tile_id"] in label_by_tile:
            unique.append((category, row))
    return unique


def generate_figures(
    *,
    index_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    label_by_tile: dict[str, dict[str, str]],
    local_figures_dir: Path,
    external_figures_dir: Path,
) -> list[str]:
    local_figures_dir.mkdir(parents=True, exist_ok=True)
    external_figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    samples_by_split: Counter[str] = Counter()
    for row in index_rows:
        for split in split_values(row["split_membership"]):
            samples_by_split[split] += 1
    figures = [
        ("samples_per_split.png", lambda p: bar_plot(dict(samples_by_split), "Samples per split", "Samples", p, external_figures_dir)),
        ("samples_per_event_location.png", lambda p: bar_plot(counts_by(index_rows, "event_location"), "Samples per event/location", "Samples", p, external_figures_dir)),
        ("status_counts.png", lambda p: bar_plot(counts_by(index_rows, "status"), "Sample status counts", "Samples", p, external_figures_dir)),
        ("cleaning_recommendation_counts.png", lambda p: bar_plot(counts_by(index_rows, "cleaning_recommendation"), "Cleaning recommendation counts", "Samples", p, external_figures_dir)),
    ]
    for filename, maker in figures:
        path = local_figures_dir / filename
        maker(path)
        generated.append(fmt_path(path))

    water_values = [to_float(row["water_pct"]) for row in label_by_tile.values()]
    invalid_values = [to_float(row["invalid_pct"]) for row in label_by_tile.values()]
    for filename, values, title, xlabel in [
        ("water_percentage_histogram.png", water_values, "Water percentage per tile", "Water pixels (%)"),
        ("invalid_percentage_histogram.png", invalid_values, "Invalid percentage per tile", "Invalid/no-data pixels (%)"),
    ]:
        path = local_figures_dir / filename
        histogram(values, title, xlabel, path, external_figures_dir)
        generated.append(fmt_path(path))

    grouped_water_split: dict[str, list[float]] = defaultdict(list)
    grouped_invalid_split: dict[str, list[float]] = defaultdict(list)
    grouped_water_event: dict[str, list[float]] = defaultdict(list)
    grouped_invalid_event: dict[str, list[float]] = defaultdict(list)
    for row in index_rows:
        label = label_by_tile.get(row["tile_id"])
        if not label:
            continue
        split = first_split(row["split_membership"])
        event = row["event_location"]
        grouped_water_split[split].append(to_float(label["water_pct"]))
        grouped_invalid_split[split].append(to_float(label["invalid_pct"]))
        grouped_water_event[event].append(to_float(label["water_pct"]))
        grouped_invalid_event[event].append(to_float(label["invalid_pct"]))

    grouped_specs = [
        ("water_percentage_by_split.png", grouped_water_split, "Water percentage by split", "Water pixels (%)", None),
        ("invalid_percentage_by_split.png", grouped_invalid_split, "Invalid percentage by split", "Invalid/no-data pixels (%)", None),
        ("water_percentage_by_event_location.png", grouped_water_event, "Water percentage by event/location", "Water pixels (%)", None),
        ("invalid_percentage_by_event_location.png", grouped_invalid_event, "Invalid percentage by event/location", "Invalid/no-data pixels (%)", None),
    ]
    for filename, grouped, title, ylabel, max_groups in grouped_specs:
        path = local_figures_dir / filename
        box_by_group(grouped, title, ylabel, path, external_figures_dir, max_groups=max_groups)
        generated.append(fmt_path(path))

    for filename, layer, title in [
        ("s1_minmax_distributions.png", "S1Hand", "Sentinel-1 min/max distributions"),
        ("s2_minmax_distributions.png", "S2Hand", "Sentinel-2 min/max distributions"),
    ]:
        path = local_figures_dir / filename
        minmax_boxplot(audit_rows, layer, title, path, external_figures_dir)
        generated.append(fmt_path(path))

    for category, row in choose_samples(index_rows, label_by_tile):
        filename = f"panel_{category}_{row['tile_id']}.png"
        path = local_figures_dir / filename
        qualitative_panel(row, label_by_tile.get(row["tile_id"]), category, path, external_figures_dir)
        generated.append(fmt_path(path))

    return generated


def build_report(
    *,
    generated_at: str,
    status: str,
    summary: dict[str, Any],
    generated_figures: list[str],
    generated_examples: list[str],
    local_report: Path,
    external_report: Path,
    local_summary_json: Path,
    external_summary_json: Path,
    blocking_reasons: list[str],
) -> str:
    global_stats = summary["global_statistics"]
    label_stats = summary["label_statistics"]
    s1_stats = summary["image_statistics"]["S1Hand"]
    s2_stats = summary["image_statistics"]["S2Hand"]
    recommendation_counts = global_stats["samples_by_cleaning_recommendation"]
    event_counts = global_stats["samples_by_event_location"]
    largest_events = sorted(event_counts.items(), key=lambda item: (-item[1], item[0]))[:5]

    lines = [
        "# STEP 3 - Sen1Floods11 statistics and visualizations report",
        "",
        "## Summary",
        f"- Status: `{status}`",
        f"- Generated at: `{generated_at}`",
        "- Raw data modified: `false`",
        "- STEP 4 started: `false`",
        "- Next step allowed: `false`",
        "",
        "## Global statistics",
        f"- Hand-labeled samples: `{global_stats['total_hand_labeled_samples']}`",
        f"- Analysis training candidates (`keep` + `warning_review`): `{global_stats['analysis_training_candidate_samples']}`",
        f"- Samples by split: `{global_stats['samples_by_split']}`",
        f"- Samples by status: `{global_stats['samples_by_status']}`",
        f"- Samples by cleaning recommendation: `{recommendation_counts}`",
        f"- No-water samples: `{global_stats['no_water_samples']}`",
        f"- High-invalid-ratio samples: `{global_stats['high_invalid_ratio_samples']}`",
        f"- Only-invalid LabelHand samples: `{global_stats['only_invalid_labelhand_samples']}`",
        f"- Largest event/location groups: `{largest_events}`",
        "",
        "## Label distribution interpretation",
        f"- Total mask pixels: `{label_stats['total_pixels']}`",
        f"- Label distribution: `{label_stats['total_label_distribution']}`",
        f"- Water percentage per tile summary: `{label_stats['water_pct_by_tile_summary']}`",
        f"- Invalid percentage per tile summary: `{label_stats['invalid_pct_by_tile_summary']}`",
        "- The hand-labeled subset is dominated by non-water pixels, so future metrics and thresholding should account for class imbalance.",
        "",
        "## Image statistics interpretation",
        f"- Sentinel-1 band counts: `{s1_stats['band_count_distribution']}`",
        f"- Sentinel-1 dtypes: `{s1_stats['dtype_distribution']}`",
        f"- Sentinel-1 NaN ratio: `{s1_stats['nan_ratio_summary']}`",
        f"- Sentinel-1 Inf ratio: `{s1_stats['inf_ratio_summary']}`",
        f"- Sentinel-1 extremes: min `{s1_stats['extreme_min']}`, max `{s1_stats['extreme_max']}`",
        f"- Sentinel-2 band counts: `{s2_stats['band_count_distribution']}`",
        f"- Sentinel-2 dtypes: `{s2_stats['dtype_distribution']}`",
        f"- Sentinel-2 NaN ratio: `{s2_stats['nan_ratio_summary']}`",
        f"- Sentinel-2 Inf ratio: `{s2_stats['inf_ratio_summary']}`",
        f"- Sentinel-2 extremes: min `{s2_stats['extreme_min']}`, max `{s2_stats['extreme_max']}`",
        "",
        "## Split/event imbalance comments",
        "- The official train/valid/test split is supplemented by a separate Bolivia holdout.",
        "- Event/location counts are uneven; validation should preserve event-aware interpretation.",
        "- The 5 fully invalid LabelHand masks are marked as `exclude_candidate` for supervised training candidates only.",
        "",
        "## Generated figures",
    ]
    lines.extend(f"- `{figure}`" for figure in generated_figures)
    lines.extend(
        [
            "",
            "## Examples generated",
        ]
    )
    lines.extend(f"- `{example}`" for example in generated_examples)
    lines.extend(
        [
            "",
            "## Recommended cleaning policy to validate later",
            "- Keep `keep` samples.",
            "- Keep `warning_review` samples for exploratory statistics and visual inspection.",
            "- Treat the 5 `exclude_candidate` samples as invalid for supervised training candidates.",
            "- Do not delete or move any raw data; any future filtering should be manifest-based.",
            "",
            "## Open questions before metrics/modeling",
            "- Should Bolivia be used only as holdout, or also for robustness reporting?",
            "- Should no-water samples remain in training to help background precision, or be balanced separately?",
            "- Should high-invalid-ratio samples be weighted, masked, or excluded in training manifests?",
            "",
            "## Generated files",
            f"- Local summary JSON: `{fmt_path(local_summary_json)}`",
            f"- External summary JSON: `{fmt_path(external_summary_json)}`",
            f"- Local report: `{fmt_path(local_report)}`",
            f"- External report: `{fmt_path(external_report)}`",
            "",
            "## Problems detected",
        ]
    )
    if blocking_reasons:
        lines.extend(f"- {reason}" for reason in blocking_reasons)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Decision required before STEP 4",
            "- Validate the cleaning policy and figure set.",
            "- Do not implement segmentation metrics until this report is reviewed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sen1floods11.yaml"))
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    local_reports_dir = repo_root / "reports"
    external_reports_dir = Path(str(config["reports"]))
    local_figures_dir = local_reports_dir / "figures"
    external_figures_dir = external_reports_dir / "figures"

    inventory_path = local_reports_dir / "sen1floods11_file_inventory.csv"
    index_path = local_reports_dir / "sen1floods11_handlabeled_index.csv"
    audit_path = local_reports_dir / "sen1floods11_handlabeled_audit.csv"
    generated_at = datetime.now().isoformat(timespec="seconds")
    blocking_reasons: list[str] = []

    for path in [inventory_path, index_path, audit_path]:
        if not path.exists():
            blocking_reasons.append(f"Missing STEP 2 input: {fmt_path(path)}")

    if blocking_reasons:
        status = "blocked"
        statistics_generated = False
        visualizations_generated = False
        summary: dict[str, Any] = {
            "generated_at": generated_at,
            "status": status,
            "blocking_reasons": blocking_reasons,
        }
        generated_figures: list[str] = []
    else:
        inventory_rows = read_csv(inventory_path)
        index_rows = read_csv(index_path)
        audit_rows = read_csv(audit_path)
        label_by_tile = label_rows_by_tile(audit_rows)
        global_stats = build_global_stats(index_rows, label_by_tile)
        label_stats = build_label_stats(index_rows, label_by_tile)
        image_stats = {
            "S1Hand": build_image_stats(audit_rows, "S1Hand"),
            "S2Hand": build_image_stats(audit_rows, "S2Hand"),
        }
        generated_figures = generate_figures(
            index_rows=index_rows,
            audit_rows=audit_rows,
            label_by_tile=label_by_tile,
            local_figures_dir=local_figures_dir,
            external_figures_dir=external_figures_dir,
        )
        statistics_generated = True
        visualizations_generated = True
        status = "done"
        summary = {
            "generated_at": generated_at,
            "status": status,
            "raw_data_modified": False,
            "input_files": {
                "inventory": fmt_path(inventory_path),
                "index": fmt_path(index_path),
                "audit": fmt_path(audit_path),
            },
            "global_statistics": global_stats,
            "label_statistics": label_stats,
            "image_statistics": image_stats,
            "generated_figures": generated_figures,
        }

    local_summary_json = local_reports_dir / "sen1floods11_step3_stats_summary.json"
    external_summary_json = external_reports_dir / "sen1floods11_step3_stats_summary.json"
    local_report = local_reports_dir / "STEP_3_statistics_visualizations_report.md"
    external_report = external_reports_dir / "STEP_3_statistics_visualizations_report.md"

    generated_examples = [
        path for path in generated_figures if Path(path).name.startswith("panel_")
    ]

    write_json(local_summary_json, summary)
    copy_to_external(local_summary_json, external_summary_json)

    report = build_report(
        generated_at=generated_at,
        status=status,
        summary=summary,
        generated_figures=generated_figures,
        generated_examples=generated_examples,
        local_report=local_report,
        external_report=external_report,
        local_summary_json=local_summary_json,
        external_summary_json=external_summary_json,
        blocking_reasons=blocking_reasons,
    )
    local_report.write_text(report, encoding="utf-8")
    copy_to_external(local_report, external_report)

    status_payload = {
        "current_step": "3",
        "status": status,
        "statistics_generated": statistics_generated,
        "visualizations_generated": visualizations_generated,
        "raw_data_modified": False,
        "summary_json": fmt_path(local_summary_json),
        "report": fmt_path(local_report),
        "figures_dir": fmt_path(local_figures_dir),
        "figure_count": len(generated_figures),
        "panel_count": len(generated_examples),
        "blocking_reasons": blocking_reasons,
        "next_step_allowed": False,
        "human_validation_required": True,
        "generated_at": generated_at,
    }
    write_json(repo_root / "pipeline_status.json", status_payload)
    write_json(Path(str(config["project_root"])) / "pipeline_status.json", status_payload)

    print("\nSTEP 3 - Sen1Floods11 statistics and visualizations summary")
    print(f"Status: {status}")
    print(f"Statistics generated: {statistics_generated}")
    print(f"Visualizations generated: {visualizations_generated}")
    print(f"Figures: {len(generated_figures)}")
    print(f"Panels: {len(generated_examples)}")
    print(f"Raw data modified: false")
    print(f"Report: {fmt_path(local_report)}")
    print("STOP: Human validation required before STEP 4.")

    return 0 if status == "done" else 2


if __name__ == "__main__":
    raise SystemExit(main())
