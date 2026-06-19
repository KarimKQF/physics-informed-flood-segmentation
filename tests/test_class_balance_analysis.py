from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts.analyze_manifest_class_balance import analyze_sample


def create_temp_raster(path: Path, data: np.ndarray) -> None:
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="+proj=latlong",
        transform=transform,
    ) as dest:
        dest.write(data, 1)


def test_analyze_sample(tmp_path: Path) -> None:
    # Create a mock manifest row
    mask_path = tmp_path / "mask.tif"

    # 2 positives, 3 negatives, 4 invalids. Total 9 pixels. Valid 5. Target rate 2/5 = 0.4
    mask_data = np.array([[1, 1, 0], [0, 0, -1], [-1, -1, -1]], dtype=np.int16)

    create_temp_raster(mask_path, mask_data)

    row = {
        "sample_id": "test_01",
        "split": "train",
        "image_path": "dummy_img.tif",
        "mask_path": str(mask_path),
        "dem_path": "dummy_dem.tif",
    }

    result = analyze_sample(row)

    assert result["sample_id"] == "test_01"
    assert result["split"] == "train"
    assert result["positive_pixel_count"] == 2
    assert result["negative_pixel_count"] == 3
    assert result["invalid_pixel_count"] == 4
    assert result["valid_pixel_count"] == 5
    assert pytest.approx(result["valid_ratio"]) == 5 / 9
    assert pytest.approx(result["target_positive_rate"]) == 2 / 5
    assert result["mask_path"] == str(mask_path)


def test_analyze_sample_empty_mask(tmp_path: Path) -> None:
    # Ensure it doesn't fail on divide by zero if no valid pixels
    mask_path = tmp_path / "mask_empty.tif"
    mask_data = np.array([[-1, -1], [-1, -1]], dtype=np.int16)
    create_temp_raster(mask_path, mask_data)

    row = {"mask_path": str(mask_path)}
    result = analyze_sample(row)

    assert result["valid_pixel_count"] == 0
    assert result["target_positive_rate"] == 0.0
