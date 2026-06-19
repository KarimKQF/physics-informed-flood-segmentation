from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

RASTER_SUFFIXES = {".tif", ".tiff"}


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"sample_id", "image_path", "mask_path", "dem_path", "split"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        return list(reader)


def raster_summary(path: Path, *, read_unique: bool = False) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Raster does not exist: {path}")
    if path.suffix.lower() not in RASTER_SUFFIXES:
        raise ValueError(f"Raster extension must be .tif or .tiff: {path}")

    with rasterio.open(path) as dataset:
        summary: dict[str, Any] = {
            "crs": dataset.crs,
            "transform": dataset.transform,
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
            "bounds": dataset.bounds,
            "res": dataset.res,
        }
        if read_unique:
            data = dataset.read(1, masked=True)
            compressed = data.compressed() if np.ma.is_masked(data) else np.asarray(data).ravel()
            summary["unique_values"] = np.unique(compressed[:1_000_000]).tolist()
        return summary


def aligned(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["crs"] == right["crs"]
        and left["width"] == right["width"]
        and left["height"] == right["height"]
        and np.allclose(tuple(left["transform"])[:6], tuple(right["transform"])[:6])
    )


def inspect_manifest(manifest_path: Path, max_samples: int) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    split_counts = Counter(row.get("split", "") for row in rows)
    inspected = 0
    problems: list[str] = []

    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Samples: {len(rows)}")
    print(f"[INFO] Split counts: {dict(split_counts)}")

    for row in rows[:max_samples]:
        inspected += 1
        sample_id = row.get("sample_id", f"sample_{inspected}")
        image_path = Path(row.get("image_path", ""))
        mask_path = Path(row.get("mask_path", ""))
        dem_value = row.get("dem_path", "").strip()
        dem_path = Path(dem_value) if dem_value else None

        print(f"\n[INFO] Sample: {sample_id}")
        try:
            image = raster_summary(image_path)
            mask = raster_summary(mask_path, read_unique=True)
            print(f"  image shape/count: ({image['height']}, {image['width']}), {image['count']}")
            print(f"  mask shape/count: ({mask['height']}, {mask['width']}), {mask['count']}")
            print(f"  image CRS: {image['crs']}")
            print(f"  mask unique values: {mask.get('unique_values', [])[:20]}")
            if not aligned(image, mask):
                problems.append(f"{sample_id}: image/mask are not aligned")
                print("  [ERROR] image/mask alignment mismatch")

            if dem_path is not None:
                dem = raster_summary(dem_path, read_unique=False)
                print(f"  dem shape/count: ({dem['height']}, {dem['width']}), {dem['count']}")
                if not aligned(image, dem):
                    problems.append(f"{sample_id}: image/DEM are not aligned")
                    print("  [ERROR] image/DEM alignment mismatch")
                with rasterio.open(dem_path) as dataset:
                    dem_data = dataset.read(1, masked=True)
                    has_nan = bool(np.isnan(np.asarray(dem_data.filled(np.nan))).any())
                    print(f"  DEM contains NaN: {has_nan}")
        except Exception as exc:
            problems.append(f"{sample_id}: {exc}")
            print(f"  [ERROR] {exc}")

    print("\n[summary]")
    print(f"inspected_samples: {inspected}")
    print(f"problems: {len(problems)}")
    for problem in problems[:20]:
        print(f"  - {problem}")

    return {
        "samples": len(rows),
        "inspected_samples": inspected,
        "split_counts": dict(split_counts),
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a GeoTIFF manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()

    try:
        result = inspect_manifest(args.manifest, args.max_samples)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0 if not result["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
