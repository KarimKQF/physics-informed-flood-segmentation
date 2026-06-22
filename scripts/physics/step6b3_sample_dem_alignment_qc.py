from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
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
    "step6b3_sample_dem_alignment_qc"
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
    "E:/flood_research/data/derived/sen1floods11_topography/"
    "dem_aligned_sample"
)
DEFAULT_DEM_SOURCE_DIR = Path("E:/flood_research/data/raw/dem/copernicus_glo30")
DEFAULT_SEED = 20260621
SPLIT_SAMPLE_COUNTS = {"train": 3, "valid": 2, "test": 2, "bolivia": 2}
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
    "topography_type",
    "crs",
    "height",
    "width",
    "finite_ratio",
    "nodata_ratio",
    "status",
    "notes",
]

QC_FIELDS = [
    "tile_id",
    "split",
    "event_location",
    "label_path",
    "source_dem_path",
    "topography_path",
    "npz_path",
    "figure_path",
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
    "min_elevation",
    "max_elevation",
    "mean_elevation",
    "std_elevation",
    "label_valid_ratio",
    "label_water_ratio",
    "plausible_elevation",
    "status",
    "notes",
]

VALIDATION_FIELDS = [
    "tile_id",
    "split",
    "event_location",
    "label_exists",
    "aligned_dem_exists",
    "tile_id_consistent",
    "crs_match",
    "shape_match",
    "transform_match",
    "finite_ratio",
    "finite_ratio_ok",
    "plausible_elevation",
    "corrupted_raster",
    "status",
    "notes",
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
            "min_elevation": None,
            "max_elevation": None,
            "mean_elevation": None,
            "std_elevation": None,
        }
    return {
        "finite_ratio": float(finite.size / total),
        "nodata_ratio": float(1.0 - finite.size / total),
        "min_elevation": float(finite.min()),
        "max_elevation": float(finite.max()),
        "mean_elevation": float(finite.mean()),
        "std_elevation": float(finite.std()),
    }


def normalize_preview(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype="float32")
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.zeros_like(array, dtype="float32")
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype("float32")


def prepare_run_dirs(run_dir: Path) -> dict[str, Path]:
    subdirs = {
        "reports": run_dir / "reports",
        "logs": run_dir / "logs",
        "scripts": run_dir / "scripts",
        "manifests": run_dir / "manifests",
        "metrics": run_dir / "metrics",
        "figures": run_dir / "figures",
        "sample_outputs": run_dir / "sample_outputs",
        "inventory": run_dir / "inventory",
        "metadata": run_dir / "metadata",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def select_samples(rows: list[dict[str, str]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for split, count in SPLIT_SAMPLE_COUNTS.items():
        candidates = [
            row
            for row in rows
            if row.get("status") == "ok"
            and row.get("split") == split
            and row.get("tile_id") not in EXCLUDED_TILE_IDS
            and Path(row.get("label_path", "")).exists()
        ]
        candidates = sorted(candidates, key=lambda item: item["tile_id"])
        if len(candidates) < count:
            raise ValueError(f"Need {count} {split} samples, found {len(candidates)}.")
        picked = rng.sample(candidates, count)
        for order, row in enumerate(sorted(picked, key=lambda item: item["tile_id"])):
            selected.append({**row, "selection_seed": seed, "selection_order": len(selected) + order})
    for index, row in enumerate(selected):
        row["selection_order"] = index
    return selected


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


def choose_dem_source(label_dataset: rasterio.DatasetReader, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
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
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def make_qc_figure(
    *,
    row: dict[str, Any],
    label_array: np.ndarray,
    dem_array: np.ndarray,
    finite_mask: np.ndarray,
    figure_path: Path,
) -> None:
    s1_preview: np.ndarray | None = None
    s1_path = Path(row.get("s1_path", ""))
    if s1_path.exists():
        try:
            with rasterio.open(s1_path) as s1:
                s1_preview = normalize_preview(s1.read(1).astype("float32"))
        except Exception:
            s1_preview = None

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    fig.suptitle(f"STEP 6B3 QC - {row['split']} - {row['tile_id']}", fontsize=12)

    label_show = np.where(np.isin(label_array, [0, 1]), label_array, np.nan)
    axes[0].imshow(label_show, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    axes[0].set_title("LabelHand")
    axes[0].axis("off")

    dem_image = axes[1].imshow(dem_array, cmap="terrain", interpolation="nearest")
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

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def align_sample(
    *,
    row: dict[str, Any],
    source: dict[str, Any],
    output_dir: Path,
    sample_outputs_dir: Path,
    figures_dir: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tile_id = row["tile_id"]
    split = row["split"]
    safe_name = f"{split}_{tile_id}_copernicus_glo30_dem"
    topography_path = output_dir / f"{safe_name}_aligned.tif"
    npz_path = sample_outputs_dir / f"{safe_name}_aligned.npz"
    metadata_path = sample_outputs_dir / f"{safe_name}_metadata.json"
    figure_path = figures_dir / f"step6b3_qc_{split}_{tile_id}.png"

    if topography_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists and --overwrite-derived was not set: {topography_path}")

    label_path = Path(row["label_path"])
    source_path = source["path"]
    with rasterio.open(label_path) as label, rasterio.open(source_path) as dem:
        destination = np.full((label.height, label.width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(dem, 1),
            destination=destination,
            src_transform=dem.transform,
            src_crs=dem.crs,
            src_nodata=dem.nodata,
            dst_transform=label.transform,
            dst_crs=label.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        label_array = label.read(1)
        stats = finite_stats(destination, np.nan)
        valid_label = np.isin(label_array, [0, 1])
        label_valid_ratio = float(valid_label.sum() / label_array.size)
        water_denominator = max(int(valid_label.sum()), 1)
        label_water_ratio = float(((label_array == 1) & valid_label).sum() / water_denominator)

        profile = label.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="lzw", BIGTIFF="IF_SAFER")
        output_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(topography_path, "w", **profile) as aligned:
            aligned.write(destination, 1)

        np.savez_compressed(
            npz_path,
            topography=destination,
            label=label_array,
            transform=np.asarray(label.transform),
            crs=str(label.crs),
            tile_id=tile_id,
            split=split,
            source_dem_path=source_path.as_posix(),
        )

        with rasterio.open(topography_path) as aligned:
            shape_ok = aligned.height == label.height and aligned.width == label.width
            crs_ok = aligned.crs == label.crs
            transform_ok = np.allclose(tuple(aligned.transform), tuple(label.transform), atol=1e-12)
            aligned_height = int(aligned.height)
            aligned_width = int(aligned.width)
            aligned_crs = str(aligned.crs)

        finite_ratio = float(stats["finite_ratio"] or 0.0)
        min_elevation = stats["min_elevation"]
        max_elevation = stats["max_elevation"]
        plausible_elevation = (
            min_elevation is not None
            and max_elevation is not None
            and -500.0 <= float(min_elevation) <= 9000.0
            and -500.0 <= float(max_elevation) <= 9000.0
        )
        status = (
            "ok"
            if shape_ok and crs_ok and transform_ok and finite_ratio > 0.95 and plausible_elevation
            else "failed_qc"
        )

        finite_mask = np.isfinite(destination)
        make_qc_figure(
            row=row,
            label_array=label_array,
            dem_array=destination,
            finite_mask=finite_mask,
            figure_path=figure_path,
        )

        manifest_row = {
            "tile_id": tile_id,
            "split": split,
            "event_location": row.get("event_location", ""),
            "s1_path": row.get("s1_path", ""),
            "s2_path": row.get("s2_path", ""),
            "label_path": label_path.as_posix(),
            "topography_path": topography_path.as_posix(),
            "topography_type": "dem_copernicus_glo30",
            "crs": str(label.crs),
            "height": int(label.height),
            "width": int(label.width),
            "finite_ratio": finite_ratio,
            "nodata_ratio": stats["nodata_ratio"],
            "status": status,
            "notes": f"source_dem={source_path.as_posix()}; npz={npz_path.as_posix()}",
        }
        qc_row = {
            "tile_id": tile_id,
            "split": split,
            "event_location": row.get("event_location", ""),
            "label_path": label_path.as_posix(),
            "source_dem_path": source_path.as_posix(),
            "topography_path": topography_path.as_posix(),
            "npz_path": npz_path.as_posix(),
            "figure_path": figure_path.as_posix(),
            "metadata_path": metadata_path.as_posix(),
            "label_crs": str(label.crs),
            "aligned_crs": aligned_crs,
            "label_height": int(label.height),
            "label_width": int(label.width),
            "aligned_height": aligned_height,
            "aligned_width": aligned_width,
            "shape_ok": shape_ok,
            "crs_ok": crs_ok,
            "transform_ok": transform_ok,
            **stats,
            "label_valid_ratio": label_valid_ratio,
            "label_water_ratio": label_water_ratio,
            "plausible_elevation": plausible_elevation,
            "status": status,
            "notes": "sample_alignment_only_not_full_dataset_validation",
        }
        validation_row = {
            "tile_id": tile_id,
            "split": split,
            "event_location": row.get("event_location", ""),
            "label_exists": label_path.exists(),
            "aligned_dem_exists": topography_path.exists(),
            "tile_id_consistent": tile_id in topography_path.name,
            "crs_match": crs_ok,
            "shape_match": shape_ok,
            "transform_match": transform_ok,
            "finite_ratio": finite_ratio,
            "finite_ratio_ok": finite_ratio > 0.95,
            "plausible_elevation": plausible_elevation,
            "corrupted_raster": status != "ok",
            "status": status,
            "notes": "",
        }
        metadata = {
            "step": "6B3",
            "generated_at": now_utc(),
            "sample": {key: row.get(key, "") for key in ["tile_id", "split", "event_location"]},
            "source_dem": {
                "cell_id": source["cell_id"],
                "path": source_path.as_posix(),
                "crs": source["crs"],
                "bounds": source["bounds"],
            },
            "outputs": {
                "topography_path": topography_path.as_posix(),
                "npz_path": npz_path.as_posix(),
                "figure_path": figure_path.as_posix(),
            },
            "qc": qc_row,
            "training_started": False,
            "raw_data_modified": False,
        }
        write_json(metadata_path, metadata)
        return manifest_row, qc_row, validation_row


def summarize_validation(validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(validation_rows)
    passed = [row for row in validation_rows if row["status"] == "ok"]
    finite_ratios = [float(row["finite_ratio"]) for row in validation_rows if row.get("finite_ratio") != ""]
    return {
        "step": "6B3",
        "generated_at": now_utc(),
        "selected_sample_count": total,
        "passed_count": len(passed),
        "failed_count": total - len(passed),
        "all_passed": total > 0 and len(passed) == total,
        "shape_match_count": sum(1 for row in validation_rows if parse_bool(row["shape_match"])),
        "crs_match_count": sum(1 for row in validation_rows if parse_bool(row["crs_match"])),
        "transform_match_count": sum(1 for row in validation_rows if parse_bool(row["transform_match"])),
        "finite_ratio_min": min(finite_ratios) if finite_ratios else None,
        "finite_ratio_threshold": 0.95,
        "plausible_elevation_count": sum(1 for row in validation_rows if parse_bool(row["plausible_elevation"])),
        "corrupted_raster_count": sum(1 for row in validation_rows if parse_bool(row["corrupted_raster"])),
        "topographic_alignment_validated": False,
        "full_topographic_alignment_completed": False,
        "training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 6B3 sample DEM alignment and QC.")
    parser.add_argument("--geospatial-inventory", type=Path, default=DEFAULT_GEOSPATIAL_INVENTORY)
    parser.add_argument("--dem-inventory", type=Path, default=DEFAULT_DEM_INVENTORY)
    parser.add_argument("--dem-source-dir", type=Path, default=DEFAULT_DEM_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite-derived", action="store_true")
    args = parser.parse_args()

    try:
        dirs = prepare_run_dirs(args.run_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        geospatial_rows = read_csv(args.geospatial_inventory)
        dem_sources = load_dem_sources(args.dem_inventory)
        selected = select_samples(geospatial_rows, args.seed)

        selected_fields = [
            "selection_order",
            "selection_seed",
            "tile_id",
            "split",
            "event_location",
            "s1_path",
            "s2_path",
            "label_path",
            "status",
        ]
        write_csv(dirs["manifests"] / "step6b3_selected_samples.csv", selected, selected_fields)
        write_json(dirs["manifests"] / "step6b3_selected_samples.json", selected)

        manifest_rows: list[dict[str, Any]] = []
        qc_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []

        for row in selected:
            with rasterio.open(row["label_path"]) as label:
                source = choose_dem_source(label, dem_sources)
            if source is None:
                validation_rows.append(
                    {
                        "tile_id": row["tile_id"],
                        "split": row["split"],
                        "event_location": row.get("event_location", ""),
                        "label_exists": Path(row["label_path"]).exists(),
                        "aligned_dem_exists": False,
                        "tile_id_consistent": False,
                        "crs_match": False,
                        "shape_match": False,
                        "transform_match": False,
                        "finite_ratio": 0.0,
                        "finite_ratio_ok": False,
                        "plausible_elevation": False,
                        "corrupted_raster": True,
                        "status": "no_intersecting_dem_tile",
                        "notes": "No Copernicus DEM tile intersects label bounds.",
                    }
                )
                continue

            manifest_row, qc_row, validation_row = align_sample(
                row=row,
                source=source,
                output_dir=args.output_dir,
                sample_outputs_dir=dirs["sample_outputs"],
                figures_dir=dirs["figures"],
                overwrite=args.overwrite_derived,
            )
            manifest_rows.append(manifest_row)
            qc_rows.append(qc_row)
            validation_rows.append(validation_row)

        write_csv(dirs["manifests"] / "topography_sample_manifest.csv", manifest_rows, MANIFEST_FIELDS)
        write_json(dirs["manifests"] / "topography_sample_manifest.json", manifest_rows)
        write_csv(dirs["metrics"] / "sample_alignment_qc.csv", qc_rows, QC_FIELDS)
        write_json(
            dirs["metrics"] / "sample_alignment_qc.json",
            {
                "step": "6B3",
                "generated_at": now_utc(),
                "dem_source": "Copernicus DEM GLO-30",
                "dem_source_dir": args.dem_source_dir.as_posix(),
                "output_dir": args.output_dir.as_posix(),
                "selected_sample_count": len(selected),
                "aligned_sample_count": len(manifest_rows),
                "qc": qc_rows,
            },
        )
        write_csv(
            dirs["metrics"] / "step6b3_geospatial_validation_details.csv",
            validation_rows,
            VALIDATION_FIELDS,
        )
        validation_summary = summarize_validation(validation_rows)
        write_json(dirs["metrics"] / "step6b3_geospatial_validation_summary.json", validation_summary)
        write_json(
            dirs["metadata"] / "step6b3_run_summary.json",
            {
                "step": "6B3",
                "generated_at": now_utc(),
                "run_dir": args.run_dir.as_posix(),
                "geospatial_inventory": args.geospatial_inventory.as_posix(),
                "dem_inventory": args.dem_inventory.as_posix(),
                "dem_source_dir": args.dem_source_dir.as_posix(),
                "derived_output_dir": args.output_dir.as_posix(),
                "selection_seed": args.seed,
                "selected_samples": len(selected),
                "aligned_samples": len(manifest_rows),
                "validation_summary": validation_summary,
                "limitations": [
                    "Copernicus DEM GLO-30 is DSM-like elevation, not HAND and not guaranteed bare-earth DTM.",
                    "Buildings and vegetation may affect local monotonic assumptions.",
                    "Resampling from about 30 m DEM pixels to Sen1Floods11 chips can smooth local relief.",
                    "Sample QC does not prove full-dataset validity.",
                    "Full 441-sample alignment remains required before physics-loss training.",
                ],
                "training_started": False,
                "raw_data_modified": False,
            },
        )

        print(f"selected_samples={len(selected)}")
        print(f"aligned_samples={len(manifest_rows)}")
        print(f"all_passed={validation_summary['all_passed']}")
        print(f"manifest={dirs['manifests'] / 'topography_sample_manifest.csv'}")
        return 0 if validation_summary["all_passed"] else 2
    except Exception as exc:  # noqa: BLE001
        error_summary = {
            "step": "6B3",
            "status": "failed_exception",
            "generated_at": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_started": False,
            "raw_data_modified": False,
        }
        DEFAULT_RUN_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
        write_json(DEFAULT_RUN_DIR / "metrics" / "step6b3_geospatial_validation_summary.json", error_summary)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
