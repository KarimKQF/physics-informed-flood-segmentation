from __future__ import annotations

import inspect
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
        config = yaml.safe_load(file) or {}
    return config


def load_dependencies():
    try:
        import geopandas as gpd
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError(
            "Missing OSM dependencies. Install them with:\n"
            "pip install -r experiments/building_masks/requirements_building_masks.txt"
        ) from exc
    return ox, gpd


def features_from_bbox(ox: Any, bbox: dict[str, Any], tags: dict[str, Any]):
    north = float(bbox["north"])
    south = float(bbox["south"])
    east = float(bbox["east"])
    west = float(bbox["west"])

    signature = inspect.signature(ox.features_from_bbox)
    params = signature.parameters

    if "bbox" in params and "north" not in params:
        attempts = (
            lambda: ox.features_from_bbox((west, south, east, north), tags=tags),
            lambda: ox.features_from_bbox((north, south, east, west), tags=tags),
        )
    else:
        attempts = (
            lambda: ox.features_from_bbox(north, south, east, west, tags=tags),
            lambda: ox.features_from_bbox((west, south, east, north), tags=tags),
        )

    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc))

    joined_errors = " | ".join(errors)
    raise RuntimeError(f"Could not call osmnx.features_from_bbox: {joined_errors}")


def fetch_buildings(config: dict[str, Any]):
    ox, _ = load_dependencies()
    ox.settings.use_cache = True

    tags = config.get("osm_tags") or {"building": True}
    query_mode = config.get("query_mode")

    if query_mode == "place":
        place_name = config.get("place_name")
        if not place_name:
            raise ValueError("query_mode is 'place' but 'place_name' is empty.")
        return ox.features_from_place(place_name, tags=tags)

    if query_mode == "bbox":
        bbox = config.get("bbox") or {}
        required_keys = {"north", "south", "east", "west"}
        missing = required_keys.difference(bbox)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"query_mode is 'bbox' but bbox is missing: {missing_keys}")
        return features_from_bbox(ox, bbox, tags)

    raise ValueError("query_mode must be either 'place' or 'bbox'.")


def keep_polygon_buildings(gdf):
    if gdf.empty or "geometry" not in gdf.columns:
        return gdf.iloc[0:0].copy()

    buildings = gdf[gdf.geometry.notna()].copy()
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if buildings.empty:
        return buildings

    buildings = buildings.reset_index()
    candidate_columns = [
        "element_type",
        "osmid",
        "building",
        "name",
        "addr:housenumber",
        "addr:street",
        "addr:city",
    ]
    columns = [column for column in candidate_columns if column in buildings.columns]
    clean = buildings[columns + ["geometry"]].copy()

    for column in columns:
        clean[column] = clean[column].where(clean[column].notna(), "").astype(str)

    return clean


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH

    try:
        config = load_config(config_path)
        buildings = keep_polygon_buildings(fetch_buildings(config))
    except Exception as exc:
        print(f"OSM building fetch failed: {exc}", file=sys.stderr)
        return 1

    if buildings.empty:
        print("No OSM building polygons were found for this query.")
        return 0

    output_config = config.get("outputs") or {}
    default_path = "outputs/geojson/osm_buildings.geojson"
    geojson_path = module_path(output_config.get("geojson_path", default_path))
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_file(geojson_path, driver="GeoJSON")

    print(f"Buildings fetched: {len(buildings)}")
    print(f"CRS: {buildings.crs}")
    print(f"Bounds: {buildings.total_bounds.tolist()}")
    print(f"Saved GeoJSON: {geojson_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
