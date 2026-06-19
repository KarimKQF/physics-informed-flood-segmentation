from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_geotiff_manifest.py"


def load_inspector_module():
    spec = importlib.util.spec_from_file_location("inspect_geotiff_manifest", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_raster(path: Path, array: np.ndarray) -> None:
    data = array[None, ...] if array.ndim == 2 else array
    profile = {
        "driver": "GTiff",
        "height": data.shape[-2],
        "width": data.shape[-1],
        "count": data.shape[0],
        "dtype": data.dtype,
        "crs": "EPSG:3857",
        "transform": from_origin(0, float(data.shape[-2]), 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data)


def test_inspect_geotiff_manifest_reports_valid_temp_manifest(tmp_path: Path) -> None:
    image_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    dem_path = tmp_path / "dem.tif"
    manifest_path = tmp_path / "manifest.csv"

    write_raster(image_path, np.ones((3, 8, 8), dtype="float32"))
    write_raster(mask_path, np.ones((8, 8), dtype="uint8"))
    write_raster(dem_path, np.ones((8, 8), dtype="float32"))

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "image_path", "mask_path", "dem_path", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample_001",
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "dem_path": dem_path.as_posix(),
                "split": "train",
            }
        )

    module = load_inspector_module()
    result = module.inspect_manifest(manifest_path, max_samples=1)

    assert result["samples"] == 1
    assert result["inspected_samples"] == 1
    assert result["split_counts"] == {"train": 1}
    assert result["problems"] == []
