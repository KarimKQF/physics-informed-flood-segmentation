from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from urban_runoff.data import GeoTIFFDataset, geotiff_collate_fn


def write_raster(path: Path, array: np.ndarray, *, transform=None) -> None:
    data = array[None, ...] if array.ndim == 2 else array
    transform = transform or from_origin(0, float(data.shape[-2]), 1, 1)
    profile = {
        "driver": "GTiff",
        "height": data.shape[-2],
        "width": data.shape[-1],
        "count": data.shape[0],
        "dtype": data.dtype,
        "crs": "EPSG:3857",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "image_path", "mask_path", "dem_path", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def create_sample(
    tmp_path: Path,
    *,
    with_dem: bool = True,
    mismatch: bool = False,
    dem_nonfinite: bool = False,
) -> Path:
    image_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    dem_path = tmp_path / "dem.tif"
    manifest_path = tmp_path / "manifest.csv"

    image = np.ones((3, 8, 8), dtype="float32")
    image[0, 0, 0] = np.nan
    write_raster(image_path, image)
    mask_shape = (7, 8) if mismatch else (8, 8)
    mask = np.zeros(mask_shape, dtype="int16")
    mask[..., 0] = -1
    mask[..., 1] = 1
    write_raster(mask_path, mask)
    if with_dem:
        dem = np.ones((8, 8), dtype="float32")
        if dem_nonfinite:
            dem[0, 0] = np.nan
            dem[0, 1] = np.inf
        write_raster(dem_path, dem)

    write_manifest(
        manifest_path,
        [
            {
                "sample_id": "sample_001",
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "dem_path": dem_path.as_posix() if with_dem else "",
                "split": "train",
            }
        ],
    )
    return manifest_path


def test_geotiff_dataset_loads_image_and_mask(tmp_path: Path) -> None:
    manifest_path = create_sample(tmp_path, with_dem=False)
    dataset = GeoTIFFDataset(manifest_path)
    sample = dataset[0]

    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (1, 8, 8)
    assert sample["valid_mask"].shape == (1, 8, 8)
    assert sample["dem"] is None
    assert torch.isfinite(sample["image"]).all()
    assert set(torch.unique(sample["mask"]).tolist()).issubset({0, 1})
    assert set(torch.unique(sample["valid_mask"]).tolist()).issubset({0, 1})
    assert sample["mask"][0, 0, 0].item() == 0
    assert sample["valid_mask"][0, 0, 0].item() == 0
    assert sample["mask"][0, 0, 1].item() == 1
    assert sample["valid_mask"][0, 0, 1].item() == 1
    assert sample["mask"][0, 0, 2].item() == 0
    assert sample["valid_mask"][0, 0, 2].item() == 1


def test_geotiff_dataset_loads_image_mask_and_dem(tmp_path: Path) -> None:
    manifest_path = create_sample(tmp_path, with_dem=True, dem_nonfinite=True)
    dataset = GeoTIFFDataset(manifest_path, require_dem=True)
    sample = dataset[0]

    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (1, 8, 8)
    assert sample["valid_mask"].shape == (1, 8, 8)
    assert sample["dem"].shape == (1, 8, 8)
    assert torch.isfinite(sample["dem"]).all()
    batch = geotiff_collate_fn([sample])
    assert batch["image"].shape == (1, 3, 8, 8)
    assert batch["mask"].shape == (1, 1, 8, 8)
    assert batch["valid_mask"].shape == (1, 1, 8, 8)
    assert batch["dem"].shape == (1, 1, 8, 8)


def test_geotiff_dataset_errors_on_shape_mismatch(tmp_path: Path) -> None:
    manifest_path = create_sample(tmp_path, with_dem=True, mismatch=True)
    dataset = GeoTIFFDataset(manifest_path)

    with pytest.raises(ValueError, match="image and mask rasters are not aligned"):
        _ = dataset[0]


def test_geotiff_dataset_errors_when_dem_required_but_absent(tmp_path: Path) -> None:
    manifest_path = create_sample(tmp_path, with_dem=False)
    dataset = GeoTIFFDataset(manifest_path, require_dem=True)

    with pytest.raises(FileNotFoundError, match="DEM is required"):
        _ = dataset[0]
