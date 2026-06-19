from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = MODULE_ROOT / "configs" / "building_osm_config.yaml"


def module_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return MODULE_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_dependencies():
    try:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        from shapely.geometry import box
    except ImportError as exc:
        raise RuntimeError(
            "Missing rasterization dependencies. Install them with:\n"
            "pip install -r experiments/building_masks/requirements_building_masks.txt"
        ) from exc
    return gpd, np, rasterio, rasterize, from_origin, box


def grid_from_reference(buildings, reference_path: Path, rasterio: Any):
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {reference_path}")

    with rasterio.open(reference_path) as reference:
        crs = reference.crs
        transform = reference.transform
        width = reference.width
        height = reference.height

    if crs is None:
        raise ValueError(f"Reference raster has no CRS: {reference_path}")

    return buildings.to_crs(crs), transform, width, height, crs


def grid_from_resolution(buildings, config: dict[str, Any], gpd: Any, from_origin: Any, box: Any):
    raster_config = config.get("raster") or {}
    resolution = float(raster_config.get("resolution_meters", 10))
    if resolution <= 0:
        raise ValueError("raster.resolution_meters must be positive.")

    source = buildings if buildings.crs else buildings.set_crs("EPSG:4326")
    target_crs = source.estimate_utm_crs()
    if target_crs is None:
        raise ValueError("Could not estimate a projected CRS for meter-based rasterization.")

    projected = source.to_crs(target_crs)
    bbox = config.get("bbox") or {}

    if {"north", "south", "east", "west"}.issubset(bbox):
        bbox_geometry = box(
            float(bbox["west"]),
            float(bbox["south"]),
            float(bbox["east"]),
            float(bbox["north"]),
        )
        bbox_bounds = gpd.GeoSeries([bbox_geometry], crs="EPSG:4326").to_crs(target_crs)
        minx, miny, maxx, maxy = bbox_bounds.total_bounds
        projected = projected.cx[minx:maxx, miny:maxy]
    else:
        minx, miny, maxx, maxy = projected.total_bounds

    width = max(1, int(math.ceil((maxx - minx) / resolution)))
    height = max(1, int(math.ceil((maxy - miny) / resolution)))
    transform = from_origin(minx, maxy, resolution, resolution)
    return projected, transform, width, height, target_crs


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH

    try:
        config = load_config(config_path)
        raster_config = config.get("raster") or {}
        if not raster_config.get("enabled", False):
            print("Rasterization is disabled in the config.")
            return 0

        gpd, np, rasterio, rasterize, from_origin, box = load_dependencies()
        outputs = config.get("outputs") or {}
        geojson_path = module_path(
            outputs.get("geojson_path", "outputs/geojson/osm_buildings.geojson")
        )
        output_path = module_path(raster_config.get("output_path", "outputs/building_mask.tif"))
        reference_raster = raster_config.get("reference_raster")

        if not geojson_path.exists():
            raise FileNotFoundError(
                f"GeoJSON not found: {geojson_path}\n"
                "Run fetch_osm_buildings.py before rasterization."
            )

        buildings = gpd.read_file(geojson_path)
        if buildings.empty:
            print("GeoJSON is empty; no raster mask was created.")
            return 0

        if reference_raster:
            reference_path = module_path(reference_raster)
            buildings, transform, width, height, crs = grid_from_reference(
                buildings, reference_path, rasterio
            )
            print(f"Raster aligned on reference raster: {reference_path}")
        else:
            buildings, transform, width, height, crs = grid_from_resolution(
                buildings, config, gpd, from_origin, box
            )
            print("No reference raster was provided.")
            print("This creates a standalone test mask only.")
            print("Later, q_i must be aligned with the imagery or DEM grid.")

        shapes = [(geometry, 1) for geometry in buildings.geometry if not geometry.is_empty]
        mask = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=bool(raster_config.get("all_touched", True)),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": np.uint8,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
            "nodata": 0,
        }

        with rasterio.open(output_path, "w", **profile) as dataset:
            dataset.write(mask, 1)
    except Exception as exc:
        print(f"OSM building rasterization failed: {exc}", file=sys.stderr)
        return 1

    print(f"Building mask saved: {output_path}")
    print(f"Shape: height={height}, width={width}")
    print("Values: 1=building, 0=non-building")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
