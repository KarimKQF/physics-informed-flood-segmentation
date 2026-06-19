from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")
rasterio_transform = pytest.importorskip("rasterio.transform")
shapely_geometry = pytest.importorskip("shapely.geometry")


REPO_ROOT = Path(__file__).resolve().parents[1]
RASTERIZE_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "building_masks"
    / "scripts"
    / "rasterize_osm_buildings_to_reference.py"
)
VALIDATE_SCRIPT = (
    REPO_ROOT / "experiments" / "building_masks" / "scripts" / "validate_building_mask_alignment.py"
)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rasterize_buildings_to_reference_creates_aligned_binary_mask(tmp_path: Path):
    rasterize_module = load_module("rasterize_osm_buildings_to_reference", RASTERIZE_SCRIPT)
    validate_module = load_module("validate_building_mask_alignment", VALIDATE_SCRIPT)

    reference_path = tmp_path / "reference.tif"
    buildings_path = tmp_path / "buildings.geojson"
    output_path = tmp_path / "building_mask_aligned.tif"

    transform = rasterio_transform.from_origin(0, 100, 10, 10)
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(reference_path, "w", **profile) as reference:
        reference.write(np.zeros((10, 10), dtype="float32"), 1)

    buildings = gpd.GeoDataFrame(
        {"building": ["yes"]},
        geometry=[shapely_geometry.box(20, 40, 60, 80)],
        crs="EPSG:3857",
    )
    buildings.to_file(buildings_path, driver="GeoJSON")

    result_path = rasterize_module.rasterize_buildings_to_reference(
        buildings_path=buildings_path,
        reference_path=reference_path,
        output_path=output_path,
        all_touched=True,
        overwrite=False,
    )

    assert result_path == output_path
    assert output_path.exists()

    with rasterio.open(reference_path) as reference, rasterio.open(output_path) as mask:
        assert mask.crs == reference.crs
        assert mask.transform == reference.transform
        assert mask.width == reference.width
        assert mask.height == reference.height
        assert mask.count == 1
        assert mask.dtypes[0] == "uint8"
        unique_values = set(np.unique(mask.read(1)).tolist())
        assert unique_values.issubset({0, 1})
        assert 1 in unique_values

    assert validate_module.validate_alignment(reference_path, output_path)
