from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")
shapely_geometry = pytest.importorskip("shapely.geometry")


REPO_ROOT = Path(__file__).resolve().parents[1]
CREATE_REFERENCE_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "building_masks"
    / "scripts"
    / "create_reference_raster_from_buildings.py"
)
RASTERIZE_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "building_masks"
    / "scripts"
    / "rasterize_osm_buildings_to_reference.py"
)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_reference_raster_from_buildings_and_rasterize_chain(tmp_path: Path):
    create_reference_module = load_module(
        "create_reference_raster_from_buildings",
        CREATE_REFERENCE_SCRIPT,
    )
    rasterize_module = load_module("rasterize_osm_buildings_to_reference", RASTERIZE_SCRIPT)

    buildings_path = tmp_path / "buildings.geojson"
    reference_path = tmp_path / "reference.tif"
    mask_path = tmp_path / "mask.tif"

    buildings = gpd.GeoDataFrame(
        {"building": ["yes"]},
        geometry=[shapely_geometry.box(0, 0, 100, 100)],
        crs="EPSG:3857",
    )
    buildings.to_file(buildings_path, driver="GeoJSON")

    create_reference_module.create_reference_raster_from_buildings(
        buildings_path=buildings_path,
        output_path=reference_path,
        target_crs="EPSG:3857",
        resolution=10,
        buffer=10,
        overwrite=False,
    )

    assert reference_path.exists()
    with rasterio.open(reference_path) as reference:
        assert reference.count == 1
        assert reference.dtypes[0] == "uint8"
        assert str(reference.crs) == "EPSG:3857"
        assert reference.width > 0
        assert reference.height > 0
        assert set(np.unique(reference.read(1)).tolist()) == {0}

    rasterize_module.rasterize_buildings_to_reference(
        buildings_path=buildings_path,
        reference_path=reference_path,
        output_path=mask_path,
        all_touched=True,
        overwrite=False,
    )

    assert mask_path.exists()
    with rasterio.open(reference_path) as reference, rasterio.open(mask_path) as mask:
        assert mask.crs == reference.crs
        assert mask.transform == reference.transform
        assert mask.width == reference.width
        assert mask.height == reference.height
        assert mask.count == 1
        assert mask.dtypes[0] == "uint8"
        unique_values = set(np.unique(mask.read(1)).tolist())
        assert unique_values.issubset({0, 1})
        assert 1 in unique_values
