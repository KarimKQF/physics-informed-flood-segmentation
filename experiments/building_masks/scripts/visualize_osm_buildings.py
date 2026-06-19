from __future__ import annotations

import json
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
        import folium
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Missing visualization dependencies. Install them with:\n"
            "pip install -r experiments/building_masks/requirements_building_masks.txt"
        ) from exc
    return folium, gpd, plt


def save_static_png(gdf, output_path: Path, plt: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, facecolor="#2d6a4f", edgecolor="#1b4332", linewidth=0.25)
    ax.set_axis_off()
    ax.set_title("OpenStreetMap building footprints", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_interactive_map(gdf, geojson_path: Path, output_path: Path, folium: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wgs84 = gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")
    west, south, east, north = wgs84.total_bounds
    center = [(south + north) / 2, (west + east) / 2]

    fmap = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
    with geojson_path.open("r", encoding="utf-8") as file:
        geojson_data = json.load(file)

    folium.GeoJson(
        geojson_data,
        name="OSM buildings",
        style_function=lambda _: {
            "fillColor": "#2d6a4f",
            "color": "#081c15",
            "weight": 0.5,
            "fillOpacity": 0.65,
        },
    ).add_to(fmap)

    fmap.fit_bounds([[south, west], [north, east]])
    folium.LayerControl().add_to(fmap)
    fmap.save(output_path)


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH

    try:
        config = load_config(config_path)
        folium, gpd, plt = load_dependencies()
        outputs = config.get("outputs") or {}
        geojson_path = module_path(
            outputs.get("geojson_path", "outputs/geojson/osm_buildings.geojson")
        )
        png_path = module_path(outputs.get("static_png_path", "outputs/figures/osm_buildings.png"))
        html_path = module_path(
            outputs.get("interactive_html_path", "outputs/maps/osm_buildings_map.html")
        )

        if not geojson_path.exists():
            raise FileNotFoundError(
                f"GeoJSON not found: {geojson_path}\n"
                "Run fetch_osm_buildings.py before visualization."
            )

        buildings = gpd.read_file(geojson_path)
        if buildings.empty:
            print("GeoJSON is empty; no visualization was created.")
            return 0

        save_static_png(buildings, png_path, plt)
        save_interactive_map(buildings, geojson_path, html_path, folium)
    except Exception as exc:
        print(f"OSM building visualization failed: {exc}", file=sys.stderr)
        return 1

    print(f"Buildings visualized: {len(buildings)}")
    print(f"Static PNG: {png_path}")
    print(f"Interactive HTML map: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
