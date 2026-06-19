from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def same_transform(first, second) -> bool:
    return np.allclose(tuple(first)[:6], tuple(second)[:6])


def same_bounds(first, second) -> bool:
    return np.allclose(
        (first.left, first.bottom, first.right, first.top),
        (second.left, second.bottom, second.right, second.top),
    )


def validate_dem_manifest(manifest_path: Path, max_samples: int) -> tuple[bool, list[str]]:
    rows = read_manifest(manifest_path)
    problems: list[str] = []

    for row in rows[:max_samples]:
        sample_id = row.get("sample_id", "unknown")
        image_path = Path(row.get("image_path", ""))
        mask_path = Path(row.get("mask_path", ""))
        dem_path = Path(row.get("dem_path", ""))

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"image_path missing: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"mask_path missing: {mask_path}")
            if not dem_path.exists():
                raise FileNotFoundError(f"dem_path missing: {dem_path}")

            with rasterio.open(image_path) as image, rasterio.open(dem_path) as dem:
                if image.crs != dem.crs:
                    raise ValueError(f"CRS mismatch: image={image.crs}, dem={dem.crs}")
                if not same_transform(image.transform, dem.transform):
                    raise ValueError("transform mismatch")
                if image.width != dem.width or image.height != dem.height:
                    raise ValueError("shape mismatch")
                if not same_bounds(image.bounds, dem.bounds):
                    raise ValueError("bounds mismatch")
                if dem.count != 1:
                    raise ValueError(f"DEM must have one band, got {dem.count}")

                data = dem.read(1)
                finite = data[np.isfinite(data)]
                nan_ratio = 1.0 - (finite.size / data.size)
                if nan_ratio > 0.01:
                    raise ValueError(f"DEM has too many NaNs/non-finite values: {nan_ratio:.4f}")
                if finite.size == 0:
                    raise ValueError("DEM has no finite values")
                dem_min = float(finite.min())
                dem_max = float(finite.max())
                if dem_min < -500 or dem_max > 9000:
                    raise ValueError(f"DEM range looks unreasonable: {dem_min} / {dem_max}")

            print(f"[OK] {sample_id} DEM aligned")
        except Exception as exc:
            message = f"{sample_id}: {exc}"
            problems.append(message)
            print(f"[ERROR] {message}")

    if problems:
        print("[FAILED] DEM alignment validation failed")
        return False, problems

    print("[OK] DEM alignment validation passed")
    return True, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DEM alignment in a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=30)
    args = parser.parse_args()

    try:
        passed, _ = validate_dem_manifest(args.manifest, args.max_samples)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
