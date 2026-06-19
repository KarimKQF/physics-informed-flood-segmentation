from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

MANIFEST_COLUMNS = ["sample_id", "image_path", "mask_path", "dem_path", "split"]


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def fill_non_finite(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.any():
        fill_value = float(np.median(values[finite]))
    else:
        fill_value = 0.0
    return np.where(finite, values, fill_value).astype("float32")


def align_dem_to_manifest(
    *,
    manifest_path: Path,
    dem_path: Path,
    output_dir: Path,
    output_manifest: Path,
    overwrite: bool,
) -> Path:
    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")
    if not dem_path.exists():
        raise FileNotFoundError(f"Raw DEM does not exist: {dem_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    updated_rows: list[dict[str, str]] = []

    with rasterio.open(dem_path) as dem:
        for row in rows:
            sample_id = row["sample_id"]
            image_path = Path(row["image_path"])
            output_path = output_dir / f"{sample_id}_DEM_aligned.tif"

            if output_path.exists() and not overwrite:
                row = dict(row)
                row["dem_path"] = output_path.as_posix()
                updated_rows.append(row)
                print(f"[OK] {sample_id} DEM already exists: {output_path}")
                continue

            with rasterio.open(image_path) as image:
                destination = np.full((image.height, image.width), np.nan, dtype="float32")
                reproject(
                    source=rasterio.band(dem, 1),
                    destination=destination,
                    src_transform=dem.transform,
                    src_crs=dem.crs,
                    src_nodata=dem.nodata,
                    dst_transform=image.transform,
                    dst_crs=image.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                destination = fill_non_finite(destination)
                profile = image.profile.copy()
                profile.update(
                    count=1,
                    dtype="float32",
                    nodata=None,
                    compress="lzw",
                )

            with rasterio.open(output_path, "w", **profile) as output:
                output.write(destination, 1)

            row = dict(row)
            row["dem_path"] = output_path.as_posix()
            updated_rows.append(row)
            print(f"[OK] {sample_id} DEM aligned: {output_path}")

    with output_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"[OK] Manifest with DEM written to: {output_manifest}")
    return output_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Align a raw DEM to each Sen1Floods11 image.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        align_dem_to_manifest(
            manifest_path=args.manifest,
            dem_path=args.dem,
            output_dir=args.output_dir,
            output_manifest=args.output_manifest,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
