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


def load_script_function(script_name: str, function_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


align_dem_to_manifest = load_script_function(
    "align_dem_to_sen1floods11_manifest.py",
    "align_dem_to_manifest",
)
validate_dem_manifest = load_script_function(
    "validate_dem_alignment_manifest.py",
    "validate_dem_manifest",
)


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


def write_manifest(path: Path, image_path: Path, mask_path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
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
                "dem_path": "",
                "split": "train",
            }
        )


def create_alignment_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    raw_dem_path = tmp_path / "raw_dem.tif"
    manifest_path = tmp_path / "manifest.csv"

    image_transform = from_origin(-65.0, -10.0, 0.01, 0.01)
    dem_transform = from_origin(-65.1, -9.9, 0.02, 0.02)
    image = np.stack(
        [
            np.ones((6, 6), dtype="float32"),
            np.full((6, 6), 2.0, dtype="float32"),
        ]
    )
    mask = np.zeros((6, 6), dtype="int16")
    mask[:, 0] = -1
    mask[:, 1:3] = 1
    raw_dem = np.linspace(100.0, 150.0, 400, dtype="float32").reshape(20, 20)
    raw_dem[0, 0] = np.nan

    write_raster(image_path, image, transform=image_transform)
    write_raster(mask_path, mask, transform=image_transform)
    write_raster(raw_dem_path, raw_dem, transform=dem_transform)
    write_manifest(manifest_path, image_path, mask_path)
    return manifest_path, raw_dem_path, image_path


def test_align_dem_writes_matching_raster_and_manifest(tmp_path: Path) -> None:
    manifest_path, raw_dem_path, image_path = create_alignment_fixture(tmp_path)
    output_dir = tmp_path / "aligned_dem"
    output_manifest = tmp_path / "manifest_with_dem.csv"

    align_dem_to_manifest(
        manifest_path=manifest_path,
        dem_path=raw_dem_path,
        output_dir=output_dir,
        output_manifest=output_manifest,
        overwrite=True,
    )

    with output_manifest.open("r", newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    aligned_dem_path = Path(row["dem_path"])
    assert aligned_dem_path.exists()

    with rasterio.open(image_path) as image, rasterio.open(aligned_dem_path) as dem:
        assert dem.crs == image.crs
        assert dem.transform == image.transform
        assert dem.width == image.width
        assert dem.height == image.height
        assert dem.count == 1
        assert dem.dtypes[0] == "float32"
        assert np.isfinite(dem.read(1)).all()


def test_validate_dem_alignment_manifest_passes(tmp_path: Path) -> None:
    manifest_path, raw_dem_path, _ = create_alignment_fixture(tmp_path)
    output_manifest = tmp_path / "manifest_with_dem.csv"
    align_dem_to_manifest(
        manifest_path=manifest_path,
        dem_path=raw_dem_path,
        output_dir=tmp_path / "aligned_dem",
        output_manifest=output_manifest,
        overwrite=True,
    )

    passed, problems = validate_dem_manifest(output_manifest, max_samples=1)

    assert passed
    assert problems == []


def test_dataloader_and_losses_work_with_aligned_dem(tmp_path: Path) -> None:
    manifest_path, raw_dem_path, _ = create_alignment_fixture(tmp_path)
    output_manifest = tmp_path / "manifest_with_dem.csv"
    align_dem_to_manifest(
        manifest_path=manifest_path,
        dem_path=raw_dem_path,
        output_dir=tmp_path / "aligned_dem",
        output_manifest=output_manifest,
        overwrite=True,
    )
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
    assert mask.shape == (1, 1, 6, 6)
    assert valid_mask.shape == (1, 1, 6, 6)
    assert dem.shape == (1, 1, 6, 6)
    assert torch.isfinite(dem).all()

    model = SimpleSegmentationCNN(in_channels=2)
    logits = model(image)
    masked_bce = MaskedBCEWithLogitsLoss()(logits, mask, valid_mask)
    topo = BinaryTopographicGradientLoss()(logits=logits, target=mask, dem=dem)
    loss = masked_bce + 0.1 * topo
    loss.backward()

    assert torch.isfinite(masked_bce)
    assert torch.isfinite(topo)
    assert torch.isfinite(loss)
