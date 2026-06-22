from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.warp import reproject, transform_bounds  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b4_full_dem_alignment"
)
DEFAULT_GEOSPATIAL_INVENTORY = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b_topographic_alignment_validation/inventory/"
    "sen1floods11_geospatial_inventory.csv"
)
DEFAULT_DEM_INVENTORY = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b2b_copernicus_dem_aws_download/inventory/"
    "copernicus_dem_verified_inventory.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "E:/flood_research/data/derived/sen1floods11_topography/dem_aligned"
)
DEFAULT_DEM_SOURCE_DIR = Path("E:/flood_research/data/raw/dem/copernicus_glo30")
DEFAULT_SEED = 20260621
EXPECTED_SPLIT_COUNTS = {"train": 251, "valid": 86, "test": 89, "bolivia": 15}
EXPECTED_TOTAL = 441
EXCLUDED_TILE_IDS = {
    "Ghana_234935",
    "Ghana_26376",
    "Ghana_277",
    "Ghana_5079",
    "Ghana_83483",
}

MANIFEST_FIELDS = [
    "tile_id",
    "split",
    "event_location",
    "s1_path",
    "s2_path",
    "label_path",
    "topography_path",
    "topography_npz_path",
    "topography_type",
    "crs",
    "height",
    "width",
    "finite_ratio",
    "nodata_ratio",
    "elevation_min",
    "elevation_max",
    "elevation_mean",
    "elevation_std",
    "status",
    "notes",
]

QC_FIELDS = [
    "tile_id",
    "split",
    "event_location",
    "s1_path",
    "s2_path",
    "label_path",
    "source_dem_path",
    "topography_path",
    "topography_npz_path",
    "metadata_path",
    "label_crs",
    "aligned_crs",
    "label_height",
    "label_width",
    "aligned_height",
    "aligned_width",
    "shape_ok",
    "crs_ok",
    "transform_ok",
    "finite_ratio",
    "nodata_ratio",
    "elevation_min",
    "elevation_max",
    "elevation_mean",
    "elevation_std",
    "label_valid_ratio",
    "label_water_ratio",
    "plausible_elevation",
    "tile_id_consistent",
    "split_consistent",
    "corrupted_raster",
    "status",
    "notes",
]

FIGURE_SAMPLE_FIELDS = [
    "tile_id",
    "split",
    "event_location",
    "topography_path",
    "label_path",
    "s1_path",
    "figure_path",
    "selection_reasons",
    "finite_ratio",
    "elevation_std",
    "elevation_min",
    "elevation_max",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def prepare_run_dirs(run_dir: Path) -> dict[str, Path]:
    subdirs = {
        "reports": run_dir / "reports",
        "logs": run_dir / "logs",
        "scripts": run_dir / "scripts",
        "manifests": run_dir / "manifests",
        "metrics": run_dir / "metrics",
        "figures": run_dir / "figures",
        "inventory": run_dir / "inventory",
        "metadata": run_dir / "metadata",
        "qc_samples": run_dir / "qc_samples",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    (subdirs["metadata"] / "per_tile").mkdir(parents=True, exist_ok=True)
    return subdirs


def bounds_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return x_overlap * y_overlap


def finite_stats(array: np.ndarray, nodata: float | None = np.nan) -> dict[str, float | None]:
    finite_mask = np.isfinite(array)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        finite_mask &= array != nodata
    finite = array[finite_mask]
    total = int(array.size)
    if finite.size == 0:
        return {
            "finite_ratio": 0.0,
            "nodata_ratio": 1.0,
            "elevation_min": None,
            "elevation_max": None,
            "elevation_mean": None,
            "elevation_std": None,
        }
    return {
        "finite_ratio": float(finite.size / total),
        "nodata_ratio": float(1.0 - finite.size / total),
        "elevation_min": float(finite.min()),
        "elevation_max": float(finite.max()),
        "elevation_mean": float(finite.mean()),
        "elevation_std": float(finite.std()),
    }


def normalize_preview(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype="float32")
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.zeros_like(array, dtype="float32")
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype("float32")


def selected_inventory_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    allowed_splits = set(EXPECTED_SPLIT_COUNTS)
    selected = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("split") in allowed_splits
        and row.get("tile_id") not in EXCLUDED_TILE_IDS
        and Path(row.get("label_path", "")).exists()
    ]
    return sorted(selected, key=lambda item: (item["split"], item["event_location"], item["tile_id"]))


def load_dem_sources(dem_inventory: Path) -> list[dict[str, Any]]:
    rows = read_csv(dem_inventory)
    sources: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["target_path"])
        if not parse_bool(row.get("exists", False)) or parse_bool(row.get("corrupt", False)):
            continue
        if not path.exists():
            continue
        sources.append(
            {
                "cell_id": row["cell_id"],
                "path": path,
                "crs": row.get("crs", ""),
                "bounds": (
                    float(row["bounds_left"]),
                    float(row["bounds_bottom"]),
                    float(row["bounds_right"]),
                    float(row["bounds_top"]),
                ),
            }
        )
    if not sources:
        raise FileNotFoundError(f"No valid DEM source rows found in {dem_inventory}")
    return sources


def choose_dem_sources(label_dataset: rasterio.DatasetReader, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_bounds = (
        label_dataset.bounds.left,
        label_dataset.bounds.bottom,
        label_dataset.bounds.right,
        label_dataset.bounds.top,
    )
    candidates: list[tuple[float, dict[str, Any]]] = []
    for source in sources:
        source_bounds = source["bounds"]
        if source["crs"] and source["crs"] != str(label_dataset.crs):
            source_bounds = transform_bounds(source["crs"], label_dataset.crs, *source_bounds, densify_pts=21)
        if bounds_intersect(source_bounds, label_bounds):
            candidates.append((intersection_area(source_bounds, label_bounds), source))
    return [source for _, source in sorted(candidates, key=lambda item: item[0], reverse=True)]


def output_paths(output_dir: Path, dirs: dict[str, Path], row: dict[str, str]) -> dict[str, Path]:
    stem = f"{row['split']}_{row['tile_id']}_copernicus_glo30_dem_aligned"
    return {
        "topography": output_dir / f"{stem}.tif",
        "npz": dirs["inventory"] / f"{stem}.npz",
        "metadata": dirs["metadata"] / "per_tile" / f"{stem}_metadata.json",
    }


def validate_existing_output(
    *,
    row: dict[str, str],
    topography_path: Path,
    npz_path: Path,
    metadata_path: Path,
    source_paths_text: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not topography_path.exists():
        return None
    try:
        with rasterio.open(row["label_path"]) as label, rasterio.open(topography_path) as aligned:
            data = aligned.read(1).astype("float32")
            label_array = label.read(1)
            stats = finite_stats(data, aligned.nodata)
            shape_ok = aligned.height == label.height and aligned.width == label.width
            crs_ok = aligned.crs == label.crs
            transform_ok = np.allclose(tuple(aligned.transform), tuple(label.transform), atol=1e-12)
            plausible = plausible_elevation(stats["elevation_min"], stats["elevation_max"])
            status = "ok" if shape_ok and crs_ok and transform_ok and (stats["finite_ratio"] or 0.0) > 0.95 and plausible else "failed_qc"
            manifest_row, qc_row = build_rows(
                row=row,
                label=label,
                label_array=label_array,
                source_paths_text=source_paths_text,
                topography_path=topography_path,
                npz_path=npz_path if npz_path.exists() else None,
                metadata_path=metadata_path,
                stats=stats,
                aligned_crs=str(aligned.crs),
                aligned_height=aligned.height,
                aligned_width=aligned.width,
                shape_ok=shape_ok,
                crs_ok=crs_ok,
                transform_ok=transform_ok,
                plausible=plausible,
                status=status,
                notes="existing_valid_output_reused" if status == "ok" else "existing_output_failed_qc",
            )
            return manifest_row, qc_row
    except Exception:
        return None


def plausible_elevation(min_value: float | None, max_value: float | None) -> bool:
    return (
        min_value is not None
        and max_value is not None
        and -500.0 <= float(min_value) <= 9000.0
        and -500.0 <= float(max_value) <= 9000.0
    )


def label_stats(label_array: np.ndarray) -> tuple[float, float]:
    valid = np.isin(label_array, [0, 1])
    valid_ratio = float(valid.sum() / label_array.size)
    denominator = max(int(valid.sum()), 1)
    water_ratio = float(((label_array == 1) & valid).sum() / denominator)
    return valid_ratio, water_ratio


def build_rows(
    *,
    row: dict[str, str],
    label: rasterio.DatasetReader,
    label_array: np.ndarray,
    source_paths_text: str,
    topography_path: Path,
    npz_path: Path | None,
    metadata_path: Path,
    stats: dict[str, float | None],
    aligned_crs: str,
    aligned_height: int,
    aligned_width: int,
    shape_ok: bool,
    crs_ok: bool,
    transform_ok: bool,
    plausible: bool,
    status: str,
    notes: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label_valid_ratio, label_water_ratio = label_stats(label_array)
    finite_ratio = float(stats["finite_ratio"] or 0.0)
    nodata_ratio = float(stats["nodata_ratio"] or 0.0)
    topography_npz_path = npz_path.as_posix() if npz_path is not None and npz_path.exists() else ""
    manifest_row = {
        "tile_id": row["tile_id"],
        "split": row["split"],
        "event_location": row.get("event_location", ""),
        "s1_path": row.get("s1_path", ""),
        "s2_path": row.get("s2_path", ""),
        "label_path": row.get("label_path", ""),
        "topography_path": topography_path.as_posix(),
        "topography_npz_path": topography_npz_path,
        "topography_type": "dem_copernicus_glo30",
        "crs": str(label.crs),
        "height": int(label.height),
        "width": int(label.width),
        "finite_ratio": finite_ratio,
        "nodata_ratio": nodata_ratio,
        "elevation_min": stats["elevation_min"],
        "elevation_max": stats["elevation_max"],
        "elevation_mean": stats["elevation_mean"],
        "elevation_std": stats["elevation_std"],
        "status": status,
        "notes": f"{notes}; source_dem={source_paths_text}",
    }
    qc_row = {
        "tile_id": row["tile_id"],
        "split": row["split"],
        "event_location": row.get("event_location", ""),
        "s1_path": row.get("s1_path", ""),
        "s2_path": row.get("s2_path", ""),
        "label_path": row.get("label_path", ""),
        "source_dem_path": source_paths_text,
        "topography_path": topography_path.as_posix(),
        "topography_npz_path": topography_npz_path,
        "metadata_path": metadata_path.as_posix(),
        "label_crs": str(label.crs),
        "aligned_crs": aligned_crs,
        "label_height": int(label.height),
        "label_width": int(label.width),
        "aligned_height": int(aligned_height),
        "aligned_width": int(aligned_width),
        "shape_ok": shape_ok,
        "crs_ok": crs_ok,
        "transform_ok": transform_ok,
        "finite_ratio": finite_ratio,
        "nodata_ratio": nodata_ratio,
        "elevation_min": stats["elevation_min"],
        "elevation_max": stats["elevation_max"],
        "elevation_mean": stats["elevation_mean"],
        "elevation_std": stats["elevation_std"],
        "label_valid_ratio": label_valid_ratio,
        "label_water_ratio": label_water_ratio,
        "plausible_elevation": plausible,
        "tile_id_consistent": row["tile_id"] in topography_path.name,
        "split_consistent": topography_path.name.startswith(f"{row['split']}_"),
        "corrupted_raster": status != "ok",
        "status": status,
        "notes": notes,
    }
    return manifest_row, qc_row


def align_one(
    *,
    row: dict[str, str],
    sources: list[dict[str, Any]],
    output_dir: Path,
    dirs: dict[str, Path],
    overwrite: bool,
    save_npz: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = output_paths(output_dir, dirs, row)
    source_paths = [source["path"] for source in sources]
    source_paths_text = ";".join(path.as_posix() for path in source_paths)
    if not overwrite:
        existing = validate_existing_output(
            row=row,
            topography_path=paths["topography"],
            npz_path=paths["npz"],
            metadata_path=paths["metadata"],
            source_paths_text=source_paths_text,
        )
        if existing is not None and existing[1]["status"] == "ok":
            return existing

    label_path = Path(row["label_path"])
    with rasterio.open(label_path) as label:
        destination = np.full((label.height, label.width), np.nan, dtype="float32")
        for source_path in source_paths:
            with rasterio.open(source_path) as dem:
                tile_buffer = np.full((label.height, label.width), np.nan, dtype="float32")
                reproject(
                    source=rasterio.band(dem, 1),
                    destination=tile_buffer,
                    src_transform=dem.transform,
                    src_crs=dem.crs,
                    src_nodata=dem.nodata,
                    dst_transform=label.transform,
                    dst_crs=label.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                valid_tile = np.isfinite(tile_buffer)
                destination[valid_tile] = tile_buffer[valid_tile]
        label_array = label.read(1)
        profile = label.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="lzw", BIGTIFF="IF_SAFER")
        paths["topography"].parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(paths["topography"], "w", **profile) as aligned:
            aligned.write(destination, 1)

        if save_npz:
            np.savez_compressed(
                paths["npz"],
                topography=destination,
                transform=np.asarray(label.transform),
                crs=str(label.crs),
                tile_id=row["tile_id"],
                split=row["split"],
                source_dem_path=source_paths_text,
            )
            npz_path: Path | None = paths["npz"]
        else:
            npz_path = None

        with rasterio.open(paths["topography"]) as aligned:
            aligned_data = aligned.read(1).astype("float32")
            stats = finite_stats(aligned_data, aligned.nodata)
            shape_ok = aligned.height == label.height and aligned.width == label.width
            crs_ok = aligned.crs == label.crs
            transform_ok = np.allclose(tuple(aligned.transform), tuple(label.transform), atol=1e-12)
            plausible = plausible_elevation(stats["elevation_min"], stats["elevation_max"])
            status = (
                "ok"
                if shape_ok and crs_ok and transform_ok and (stats["finite_ratio"] or 0.0) > 0.95 and plausible
                else "failed_qc"
            )
            manifest_row, qc_row = build_rows(
                row=row,
                label=label,
                label_array=label_array,
                source_paths_text=source_paths_text,
                topography_path=paths["topography"],
                npz_path=npz_path,
                metadata_path=paths["metadata"],
                stats=stats,
                aligned_crs=str(aligned.crs),
                aligned_height=aligned.height,
                aligned_width=aligned.width,
                shape_ok=shape_ok,
                crs_ok=crs_ok,
                transform_ok=transform_ok,
                plausible=plausible,
                status=status,
                notes="aligned_full_dataset",
            )

    metadata = {
        "step": "6B4",
        "generated_at": now_utc(),
        "sample": {key: row.get(key, "") for key in ["tile_id", "split", "event_location"]},
        "source_dem_tiles": [
            {
                "cell_id": source["cell_id"],
                "path": source["path"].as_posix(),
                "crs": source["crs"],
                "bounds": source["bounds"],
            }
            for source in sources
        ],
        "outputs": {
            "topography_path": paths["topography"].as_posix(),
            "topography_npz_path": paths["npz"].as_posix() if save_npz else "",
        },
        "qc": qc_row,
        "training_started": False,
        "raw_data_modified": False,
    }
    write_json(paths["metadata"], metadata)
    return manifest_row, qc_row


def make_qc_figure(qc_row: dict[str, Any], figure_path: Path) -> None:
    label_path = Path(qc_row["label_path"])
    topography_path = Path(qc_row["topography_path"])
    s1_path = Path(qc_row.get("s1_path", ""))
    with rasterio.open(label_path) as label_ds:
        label_array = label_ds.read(1)
    with rasterio.open(topography_path) as topo_ds:
        topo_array = topo_ds.read(1).astype("float32")
    finite_mask = np.isfinite(topo_array)

    s1_preview: np.ndarray | None = None
    if s1_path.exists():
        try:
            with rasterio.open(s1_path) as s1:
                s1_preview = normalize_preview(s1.read(1).astype("float32"))
        except Exception:
            s1_preview = None

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    fig.suptitle(f"STEP 6B4 QC - {qc_row['split']} - {qc_row['tile_id']}", fontsize=12)

    label_show = np.where(np.isin(label_array, [0, 1]), label_array, np.nan)
    axes[0].imshow(label_show, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    axes[0].set_title("LabelHand")
    axes[0].axis("off")

    dem_image = axes[1].imshow(topo_array, cmap="terrain", interpolation="nearest")
    axes[1].set_title("Aligned DEM")
    axes[1].axis("off")
    fig.colorbar(dem_image, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(finite_mask, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axes[2].set_title("Finite DEM mask")
    axes[2].axis("off")

    if s1_preview is not None:
        axes[3].imshow(s1_preview, cmap="gray", interpolation="nearest")
        axes[3].set_title("S1 VV preview")
    else:
        axes[3].imshow(np.zeros_like(label_array), cmap="gray", interpolation="nearest")
        axes[3].set_title("S1 preview unavailable")
    axes[3].axis("off")

    finite_values = topo_array[np.isfinite(topo_array)]
    axes[4].hist(finite_values, bins=40, color="#4974a5")
    axes[4].set_title("Elevation histogram")
    axes[4].set_xlabel("Elevation")
    axes[4].set_ylabel("Pixels")

    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.95,
        "\n".join(
            [
                f"finite_ratio: {float(qc_row['finite_ratio']):.4f}",
                f"min: {float(qc_row['elevation_min']):.2f}",
                f"max: {float(qc_row['elevation_max']):.2f}",
                f"std: {float(qc_row['elevation_std']):.2f}",
                f"event: {qc_row['event_location']}",
            ]
        ),
        va="top",
        fontsize=10,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=145)
    plt.close(fig)


def select_qc_figure_samples(qc_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    ok_rows = [row for row in qc_rows if row["status"] == "ok"]
    by_key = {row["tile_id"]: {**row, "selection_reasons": []} for row in ok_rows}
    rng = random.Random(seed)

    def add(row: dict[str, Any], reason: str) -> None:
        by_key[row["tile_id"]]["selection_reasons"].append(reason)

    for split in EXPECTED_SPLIT_COUNTS:
        candidates = sorted([row for row in ok_rows if row["split"] == split], key=lambda item: item["tile_id"])
        for row in rng.sample(candidates, min(2, len(candidates))):
            add(row, f"split_{split}")

    for event_location in sorted({row["event_location"] for row in ok_rows}):
        candidates = sorted(
            [row for row in ok_rows if row["event_location"] == event_location],
            key=lambda item: item["tile_id"],
        )
        if candidates:
            add(rng.choice(candidates), f"event_{event_location}")

    relief_sorted = sorted(ok_rows, key=lambda item: float(item["elevation_std"]))
    for row in relief_sorted[:3]:
        add(row, "low_relief")
    for row in relief_sorted[-3:]:
        add(row, "high_relief")

    selected = [
        row
        for row in by_key.values()
        if row["selection_reasons"]
    ]
    return sorted(selected, key=lambda item: (item["split"], item["event_location"], item["tile_id"]))


def generate_qc_figures(qc_rows: list[dict[str, Any]], dirs: dict[str, Path], seed: int) -> list[dict[str, Any]]:
    selected = select_qc_figure_samples(qc_rows, seed)
    figure_rows: list[dict[str, Any]] = []
    for row in selected:
        figure_path = dirs["figures"] / f"step6b4_qc_{row['split']}_{row['tile_id']}.png"
        make_qc_figure(row, figure_path)
        figure_rows.append(
            {
                "tile_id": row["tile_id"],
                "split": row["split"],
                "event_location": row["event_location"],
                "topography_path": row["topography_path"],
                "label_path": row["label_path"],
                "s1_path": row.get("s1_path", ""),
                "figure_path": figure_path.as_posix(),
                "selection_reasons": ";".join(sorted(set(row["selection_reasons"]))),
                "finite_ratio": row["finite_ratio"],
                "elevation_std": row["elevation_std"],
                "elevation_min": row["elevation_min"],
                "elevation_max": row["elevation_max"],
            }
        )
    write_csv(dirs["manifests"] / "step6b4_qc_figure_samples.csv", figure_rows, FIGURE_SAMPLE_FIELDS)
    write_json(dirs["manifests"] / "step6b4_qc_figure_samples.json", figure_rows)
    return figure_rows


def count_nested(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[str(row[key])][str(row["status"])] += 1
    return {group: dict(counter) for group, counter in sorted(grouped.items())}


def summarize_alignment(
    *,
    selected_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    figure_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    finite_ratios = [float(row["finite_ratio"]) for row in qc_rows if row.get("finite_ratio") not in {"", None}]
    ok_rows = [row for row in qc_rows if row["status"] == "ok"]
    split_counts = Counter(row["split"] for row in selected_rows)
    split_pass_counts = Counter(row["split"] for row in ok_rows)
    event_counts = Counter(row["event_location"] for row in selected_rows)
    event_pass_counts = Counter(row["event_location"] for row in ok_rows)
    all_core_checks = (
        len(selected_rows) == EXPECTED_TOTAL
        and len(manifest_rows) == EXPECTED_TOTAL
        and len(ok_rows) == EXPECTED_TOTAL
        and all(parse_bool(row["shape_ok"]) for row in qc_rows)
        and all(parse_bool(row["crs_ok"]) for row in qc_rows)
        and all(parse_bool(row["transform_ok"]) for row in qc_rows)
        and all(float(row["finite_ratio"]) > 0.95 for row in qc_rows)
    )
    return {
        "step": "6B4",
        "generated_at": now_utc(),
        "expected_samples": EXPECTED_TOTAL,
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "input_selected_samples": len(selected_rows),
        "aligned_samples": len(manifest_rows),
        "valid_outputs": len(ok_rows),
        "missing_outputs": sum(1 for row in qc_rows if not Path(row["topography_path"]).exists()),
        "failed_outputs": len(qc_rows) - len(ok_rows),
        "shape_pass_count": sum(1 for row in qc_rows if parse_bool(row["shape_ok"])),
        "crs_pass_count": sum(1 for row in qc_rows if parse_bool(row["crs_ok"])),
        "transform_pass_count": sum(1 for row in qc_rows if parse_bool(row["transform_ok"])),
        "finite_ratio_min": min(finite_ratios) if finite_ratios else None,
        "finite_ratio_mean": float(np.mean(finite_ratios)) if finite_ratios else None,
        "finite_ratio_threshold": 0.95,
        "corrupted_count": sum(1 for row in qc_rows if parse_bool(row["corrupted_raster"])),
        "split_level_counts": {
            split: {"expected": EXPECTED_SPLIT_COUNTS[split], "selected": split_counts[split], "passed": split_pass_counts[split]}
            for split in EXPECTED_SPLIT_COUNTS
        },
        "event_location_level_counts": {
            event: {"selected": event_counts[event], "passed": event_pass_counts[event]}
            for event in sorted(event_counts)
        },
        "status_counts": dict(Counter(row["status"] for row in qc_rows)),
        "qc_figure_count": len(figure_rows),
        "output_dir": output_dir.as_posix(),
        "full_topographic_alignment_completed": len(manifest_rows) == EXPECTED_TOTAL,
        "full_topographic_alignment_qc_passed": all_core_checks,
        "topography_full_manifest_created": len(manifest_rows) == EXPECTED_TOTAL,
        "topographic_alignment_validated": False,
        "training_started": False,
        "raw_data_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 6B4 full Copernicus DEM alignment for Sen1Floods11.")
    parser.add_argument("--geospatial-inventory", type=Path, default=DEFAULT_GEOSPATIAL_INVENTORY)
    parser.add_argument("--dem-inventory", type=Path, default=DEFAULT_DEM_INVENTORY)
    parser.add_argument("--dem-source-dir", type=Path, default=DEFAULT_DEM_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-npz", action="store_true")
    args = parser.parse_args()

    dirs = prepare_run_dirs(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        geospatial_rows = read_csv(args.geospatial_inventory)
        selected_rows = selected_inventory_rows(geospatial_rows)
        dem_sources = load_dem_sources(args.dem_inventory)
        input_split_counts = dict(Counter(row["split"] for row in selected_rows))
        write_json(
            dirs["metadata"] / "step6b4_input_summary.json",
            {
                "step": "6B4",
                "generated_at": now_utc(),
                "geospatial_inventory": args.geospatial_inventory.as_posix(),
                "dem_inventory": args.dem_inventory.as_posix(),
                "dem_source_dir": args.dem_source_dir.as_posix(),
                "output_dir": args.output_dir.as_posix(),
                "selected_samples": len(selected_rows),
                "expected_samples": EXPECTED_TOTAL,
                "input_split_counts": input_split_counts,
                "expected_split_counts": EXPECTED_SPLIT_COUNTS,
                "excluded_tile_ids": sorted(EXCLUDED_TILE_IDS),
            },
        )
        if len(selected_rows) != EXPECTED_TOTAL:
            raise ValueError(f"Expected {EXPECTED_TOTAL} selected rows, found {len(selected_rows)}.")

        manifest_rows: list[dict[str, Any]] = []
        qc_rows: list[dict[str, Any]] = []
        for index, row in enumerate(selected_rows, start=1):
            with rasterio.open(row["label_path"]) as label:
                sources = choose_dem_sources(label, dem_sources)
            if not sources:
                paths = output_paths(args.output_dir, dirs, row)
                qc_rows.append(
                    {
                        "tile_id": row["tile_id"],
                        "split": row["split"],
                        "event_location": row.get("event_location", ""),
                        "label_path": row["label_path"],
                        "source_dem_path": "",
                        "topography_path": paths["topography"].as_posix(),
                        "topography_npz_path": "",
                        "metadata_path": paths["metadata"].as_posix(),
                        "label_crs": "",
                        "aligned_crs": "",
                        "label_height": "",
                        "label_width": "",
                        "aligned_height": "",
                        "aligned_width": "",
                        "shape_ok": False,
                        "crs_ok": False,
                        "transform_ok": False,
                        "finite_ratio": 0.0,
                        "nodata_ratio": 1.0,
                        "elevation_min": "",
                        "elevation_max": "",
                        "elevation_mean": "",
                        "elevation_std": "",
                        "label_valid_ratio": "",
                        "label_water_ratio": "",
                        "plausible_elevation": False,
                        "tile_id_consistent": False,
                        "split_consistent": False,
                        "corrupted_raster": True,
                        "status": "no_intersecting_dem_tile",
                        "notes": "No verified Copernicus DEM tile intersects label bounds.",
                    }
                )
                print(f"[{index}/{len(selected_rows)}] {row['tile_id']} no_intersecting_dem_tile", flush=True)
                continue
            manifest_row, qc_row = align_one(
                row=row,
                sources=sources,
                output_dir=args.output_dir,
                dirs=dirs,
                overwrite=args.overwrite,
                save_npz=args.save_npz,
            )
            manifest_rows.append(manifest_row)
            qc_rows.append(qc_row)
            print(f"[{index}/{len(selected_rows)}] {row['tile_id']} {qc_row['status']}", flush=True)

        figure_rows = generate_qc_figures(qc_rows, dirs, args.seed)
        summary = summarize_alignment(
            selected_rows=selected_rows,
            manifest_rows=manifest_rows,
            qc_rows=qc_rows,
            figure_rows=figure_rows,
            output_dir=args.output_dir,
        )

        write_csv(dirs["manifests"] / "topography_full_manifest.csv", manifest_rows, MANIFEST_FIELDS)
        write_json(dirs["manifests"] / "topography_full_manifest.json", manifest_rows)
        write_csv(dirs["metrics"] / "step6b4_full_alignment_qc.csv", qc_rows, QC_FIELDS)
        write_json(
            dirs["metrics"] / "step6b4_full_alignment_qc.json",
            {"summary": summary, "qc": qc_rows},
        )
        write_json(dirs["metrics"] / "step6b4_full_alignment_summary.json", summary)

        print(f"expected_samples={summary['expected_samples']}")
        print(f"aligned_samples={summary['aligned_samples']}")
        print(f"failed_outputs={summary['failed_outputs']}")
        print(f"finite_ratio_min={summary['finite_ratio_min']}")
        print(f"full_topographic_alignment_qc_passed={summary['full_topographic_alignment_qc_passed']}")
        print(f"manifest={dirs['manifests'] / 'topography_full_manifest.csv'}")
        return 0 if summary["full_topographic_alignment_qc_passed"] else 2
    except Exception as exc:  # noqa: BLE001
        failure = {
            "step": "6B4",
            "generated_at": now_utc(),
            "status": "failed_exception",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "full_topographic_alignment_qc_passed": False,
            "training_started": False,
            "raw_data_modified": False,
        }
        write_json(dirs["metrics"] / "step6b4_full_alignment_summary.json", failure)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
