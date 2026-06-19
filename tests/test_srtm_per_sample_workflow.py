from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

from urban_runoff.data import GeoTIFFDataset, geotiff_collate_fn
from urban_runoff.losses import BinaryTopographicGradientLoss, MaskedBCEWithLogitsLoss
from urban_runoff.models import SimpleSegmentationCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(script_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compact_module = load_script_module("create_compact_sen1floods11_subset_manifest.py")
srtm_module = load_script_module("download_and_align_srtm_per_sample.py")
validation_module = load_script_module("validate_dem_alignment_manifest.py")


def write_raster(path: Path, array: np.ndarray, *, transform) -> None:
    data = array[None, ...] if array.ndim == 2 else array
    profile = {
        "driver": "GTiff",
        "height": data.shape[-2],
        "width": data.shape[-1],
        "count": data.shape[0],
        "dtype": data.dtype,
        "crs": "EPSG:4326",
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


def test_srtm_tile_names_for_simple_bounds() -> None:
    bounds = (-65.2, -14.2, -65.1, -14.1)

    tiles = srtm_module.srtm_tile_names(bounds)

    assert tiles == ["S15W066"]


def test_compact_subset_prefers_samples_with_shared_tiles() -> None:
    rows = [
        {
            "sample_id": "a",
            "image_path": "a.tif",
            "mask_path": "a_mask.tif",
            "dem_path": "",
            "split": "train",
        },
        {
            "sample_id": "b",
            "image_path": "b.tif",
            "mask_path": "b_mask.tif",
            "dem_path": "",
            "split": "train",
        },
        {
            "sample_id": "c",
            "image_path": "c.tif",
            "mask_path": "c_mask.tif",
            "dem_path": "",
            "split": "train",
        },
    ]
    bounds_by_sample = {
        "a": {"bounds": {"left": -65.2, "bottom": -14.2, "right": -65.1, "top": -14.1}},
        "b": {"bounds": {"left": -65.3, "bottom": -14.3, "right": -65.2, "top": -14.2}},
        "c": {"bounds": {"left": -10.3, "bottom": 1.2, "right": -10.2, "top": 1.3}},
    }

    selected_rows, unique_tiles = compact_module.select_compact_samples(
        rows,
        bounds_by_sample,
        max_samples=2,
        max_unique_srtm_tiles=1,
    )

    assert [row["sample_id"] for row in selected_rows] == ["a", "b"]
    assert unique_tiles == ["S15W066"]
    assert all(row["dem_path"] == "" for row in selected_rows)


def create_sample_manifest(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    manifest_path = tmp_path / "manifest.csv"

    image_transform = from_origin(-65.2, -14.1, 0.01, 0.01)
    image = np.stack(
        [
            np.ones((6, 6), dtype="float32"),
            np.full((6, 6), 2.0, dtype="float32"),
        ]
    )
    mask = np.zeros((6, 6), dtype="int16")
    mask[:, 0] = -1
    mask[:, 1:3] = 1
    write_raster(image_path, image, transform=image_transform)
    write_raster(mask_path, mask, transform=image_transform)
    write_manifest(
        manifest_path,
        [
            {
                "sample_id": "sample_001",
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "dem_path": "",
                "split": "train",
            }
        ],
    )
    return manifest_path, image_path


def create_mock_srtm_tile(tmp_path: Path) -> Path:
    tile_path = tmp_path / "S15W066_mock.tif"
    transform = from_origin(-66.0, -14.0, 0.01, 0.01)
    data = np.linspace(100.0, 200.0, 10000, dtype="float32").reshape(100, 100)
    write_raster(tile_path, data, transform=transform)
    return tile_path


def test_download_and_align_per_sample_with_mocked_tile(tmp_path: Path, monkeypatch) -> None:
    manifest_path, _ = create_sample_manifest(tmp_path)
    tile_path = create_mock_srtm_tile(tmp_path)
    output_manifest = tmp_path / "manifest_with_dem.csv"

    def fake_download(tile_name: str, tile_cache_dir: Path):
        assert tile_name == "S15W066"
        assert tile_cache_dir == tmp_path / "cache"
        return tile_path, True, "mock://S15W066"

    monkeypatch.setattr(srtm_module, "download_or_reuse_srtm_tile", fake_download)

    summary = srtm_module.download_and_align(
        manifest_path=manifest_path,
        output_manifest=output_manifest,
        tile_cache_dir=tmp_path / "cache",
        aligned_dem_dir=tmp_path / "aligned_dem",
        max_samples=1,
        max_tiles_total=1,
        overwrite=True,
        buffer_degrees=0.0,
    )

    assert summary["unique_tiles"] == ["S15W066"]
    assert summary["downloaded_tiles"] == ["S15W066"]
    assert output_manifest.exists()

    passed, problems = validation_module.validate_dem_manifest(output_manifest, max_samples=1)
    assert passed
    assert problems == []

    dataset = GeoTIFFDataset(output_manifest, require_dem=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=geotiff_collate_fn)))
    image = batch["image"]
    mask = batch["mask"]
    valid_mask = batch["valid_mask"]
    dem = batch["dem"]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(valid_mask, torch.Tensor)
    assert isinstance(dem, torch.Tensor)
    assert image.shape == (1, 2, 6, 6)
    assert dem.shape == (1, 1, 6, 6)
    assert torch.isfinite(dem).all()

    model = SimpleSegmentationCNN(in_channels=2)
    logits = model(image)
    masked_bce = MaskedBCEWithLogitsLoss()(logits, mask, valid_mask)
    topographic = BinaryTopographicGradientLoss()(logits=logits, target=mask, dem=dem)
    total = masked_bce + 0.1 * topographic
    total.backward()

    assert torch.isfinite(masked_bce)
    assert torch.isfinite(topographic)
    assert torch.isfinite(total)
