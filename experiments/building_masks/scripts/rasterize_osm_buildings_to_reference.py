from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_ok(message: str) -> None:
    print(f"[OK] {message}")


def ensure_input_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def filter_valid_building_geometries(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "geometry" not in buildings.columns:
        raise ValueError("Buildings GeoJSON has no geometry column.")

    filtered = buildings[buildings.geometry.notna()].copy()
    filtered = filtered[~filtered.geometry.is_empty].copy()
    filtered = filtered[filtered.geometry.is_valid].copy()
    filtered = filtered[filtered.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return filtered


def rasterize_buildings_to_reference(
    buildings_path: Path,
    reference_path: Path,
    output_path: Path,
    all_touched: bool = False,
    overwrite: bool = False,
) -> Path:
    buildings_path = Path(buildings_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)

    ensure_input_file(buildings_path, "Buildings GeoJSON")
    ensure_input_file(reference_path, "Reference raster")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_info(f"Loading reference raster: {reference_path}")
    with rasterio.open(reference_path) as reference:
        reference_crs = reference.crs
        reference_transform = reference.transform
        reference_width = reference.width
        reference_height = reference.height
        reference_bounds = reference.bounds
        reference_resolution = reference.res
        reference_profile = reference.profile.copy()

    if reference_crs is None:
        raise ValueError(f"Reference raster has no CRS: {reference_path}")

    log_info(f"Reference CRS: {reference_crs}")
    log_info(f"Reference shape: height={reference_height}, width={reference_width}")
    log_info(f"Reference bounds: {reference_bounds}")
    log_info(f"Reference resolution: {reference_resolution}")

    log_info(f"Loading buildings: {buildings_path}")
    buildings = gpd.read_file(buildings_path)
    if buildings.empty:
        raise ValueError(f"Buildings GeoJSON is empty: {buildings_path}")
    if buildings.crs is None:
        raise ValueError(f"Buildings GeoJSON has no CRS: {buildings_path}")

    log_info(f"Buildings count before filtering: {len(buildings)}")

    if buildings.crs != reference_crs:
        log_info("Reprojecting buildings to reference CRS...")
        buildings = buildings.to_crs(reference_crs)

    buildings = filter_valid_building_geometries(buildings)
    log_info(f"Buildings count after filtering: {len(buildings)}")

    if buildings.empty:
        raise ValueError("No valid Polygon or MultiPolygon building geometries remain.")

    shapes = [(geometry, 1) for geometry in buildings.geometry]
    log_info("Rasterizing buildings...")
    mask = rasterize(
        shapes=shapes,
        out_shape=(reference_height, reference_width),
        transform=reference_transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=all_touched,
    )
    mask = np.where(mask > 0, 1, 0).astype("uint8")

    profile = reference_profile
    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw",
    )

    log_info(f"Writing aligned building mask to: {output_path}")
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(mask, 1)

    log_ok("Building mask successfully created.")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rasterize OSM building polygons on a reference raster grid."
    )
    parser.add_argument(
        "--buildings",
        required=True,
        type=Path,
        help="Path to the OSM buildings GeoJSON.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Path to the reference GeoTIFF raster.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the aligned building mask GeoTIFF will be written.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Rasterize every pixel touched by a building polygon.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output mask if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rasterize_buildings_to_reference(
            buildings_path=args.buildings,
            reference_path=args.reference,
            output_path=args.output,
            all_touched=args.all_touched,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
