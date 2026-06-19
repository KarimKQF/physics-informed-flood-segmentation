from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_ok(message: str) -> None:
    print(f"[OK] {message}")


def ensure_input_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def create_reference_raster_from_buildings(
    buildings_path: Path,
    output_path: Path,
    target_crs: str = "EPSG:2154",
    resolution: float = 10.0,
    buffer: float = 100.0,
    overwrite: bool = False,
) -> Path:
    buildings_path = Path(buildings_path)
    output_path = Path(output_path)

    ensure_input_file(buildings_path, "Buildings GeoJSON")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    if resolution <= 0:
        raise ValueError("Resolution must be positive.")
    if buffer < 0:
        raise ValueError("Buffer must be greater than or equal to zero.")

    log_info(f"Loading buildings: {buildings_path}")
    buildings = gpd.read_file(buildings_path)
    if buildings.empty:
        raise ValueError(f"Buildings GeoJSON is empty: {buildings_path}")
    if buildings.crs is None:
        raise ValueError(f"Buildings GeoJSON has no CRS: {buildings_path}")

    log_info(f"Buildings CRS: {buildings.crs}")
    log_info(f"Reprojecting buildings to {target_crs}...")
    projected = buildings.to_crs(target_crs)

    minx, miny, maxx, maxy = projected.total_bounds
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer
    log_info(f"Bounds with buffer: [{minx}, {miny}, {maxx}, {maxy}]")
    log_info(f"Resolution: {resolution:g} meters")

    width = max(1, int(math.ceil((maxx - minx) / resolution)))
    height = max(1, int(math.ceil((maxy - miny) / resolution)))
    transform = from_origin(minx, maxy, resolution, resolution)
    log_info(f"Raster shape: height={height}, width={width}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": target_crs,
        "transform": transform,
        "nodata": 0,
        "compress": "lzw",
    }

    log_info(f"Writing reference raster to: {output_path}")
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(np.zeros((height, width), dtype="uint8"), 1)

    log_ok("Reference raster successfully created.")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a georeferenced development raster from OSM building bounds."
    )
    parser.add_argument(
        "--buildings",
        required=True,
        type=Path,
        help="Path to the OSM buildings GeoJSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the reference GeoTIFF will be written.",
    )
    parser.add_argument(
        "--target-crs",
        default="EPSG:2154",
        help="Target CRS for the reference raster. Default: EPSG:2154.",
    )
    parser.add_argument(
        "--resolution",
        default=10.0,
        type=float,
        help="Raster resolution in meters. Default: 10.",
    )
    parser.add_argument(
        "--buffer",
        default=100.0,
        type=float,
        help="Buffer around building bounds in target CRS meters. Default: 100.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output raster if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        create_reference_raster_from_buildings(
            buildings_path=args.buildings,
            output_path=args.output,
            target_crs=args.target_crs,
            resolution=args.resolution,
            buffer=args.buffer,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
