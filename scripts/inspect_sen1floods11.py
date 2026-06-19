from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np

MASK_KEYWORDS = ("label", "mask", "hand", "water", "flood")
RASTER_SUFFIXES = (".tif", ".tiff")


def _load_rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise SystemExit(
            "rasterio is required to inspect GeoTIFF metadata. "
            'Install dependencies with: pip install -e ".[dev]"'
        ) from exc
    return rasterio


def _find_rasters(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES
    )


def _is_mask_candidate(path: Path) -> bool:
    lowered = str(path).lower()
    return any(keyword in lowered for keyword in MASK_KEYWORDS)


def _format_examples(paths: Iterable[Path], root: Path, limit: int) -> None:
    for path in list(paths)[:limit]:
        print(f"  - {path.relative_to(root)}")


def _read_raster_summary(path: Path, root: Path, show_unique: bool) -> dict[str, object]:
    rasterio = _load_rasterio()
    summary: dict[str, object] = {"shape": None, "unique_values": []}
    print(f"\n{path.relative_to(root)}")
    try:
        with rasterio.open(path) as dataset:
            band = dataset.read(1, masked=True)
            compressed = band.compressed() if np.ma.is_masked(band) else np.asarray(band).ravel()
            finite = compressed[np.isfinite(compressed)] if compressed.size else compressed
            summary["shape"] = (dataset.height, dataset.width)
            print(f"  shape: ({dataset.height}, {dataset.width}), count: {dataset.count}")
            print(f"  crs: {dataset.crs}")
            print(f"  transform: {dataset.transform}")
            print(f"  dtype: {dataset.dtypes[0] if dataset.dtypes else 'unknown'}")
            if finite.size:
                print(f"  min/max: {float(finite.min()):.6g} / {float(finite.max()):.6g}")
            else:
                print("  min/max: no finite values")
            if show_unique:
                unique = np.unique(compressed[:1_000_000]) if compressed.size else np.array([])
                summary["unique_values"] = [str(value) for value in unique[:30].tolist()]
                print(f"  unique sample values: {unique[:30].tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read raster: {exc}")
    return summary


def inspect_dataset(root: Path, max_rasters: int, max_masks: int) -> None:
    if not root.exists():
        raise SystemExit(
            f"{root} does not exist. Download first with: bash scripts/download_sen1floods11.sh"
        )

    files = sorted(path for path in root.rglob("*") if path.is_file())
    subdirs = sorted(path for path in root.iterdir() if path.is_dir())
    rasters = _find_rasters(root)
    mask_candidates = [path for path in rasters if _is_mask_candidate(path)]
    image_candidates = [path for path in rasters if not _is_mask_candidate(path)]
    observed_shapes: set[tuple[int, int]] = set()
    observed_label_values: set[str] = set()

    print(f"Root: {root}")
    print(f"Total files: {len(files)}")
    print(f"Subdirectories: {len(subdirs)}")
    _format_examples(subdirs, root, limit=10)
    print(f"GeoTIFF files: {len(rasters)}")
    print("Example GeoTIFF paths:")
    _format_examples(rasters, root, limit=10)
    print(f"Mask-like GeoTIFF candidates: {len(mask_candidates)}")
    print("Example mask-like paths:")
    _format_examples(mask_candidates, root, limit=10)

    print("\nRaster metadata samples:")
    for path in rasters[:max_rasters]:
        summary = _read_raster_summary(path, root, show_unique=_is_mask_candidate(path))
        shape = summary["shape"]
        if isinstance(shape, tuple):
            observed_shapes.add(shape)

    if mask_candidates:
        print("\nMask candidate value samples:")
        for path in mask_candidates[:max_masks]:
            summary = _read_raster_summary(path, root, show_unique=True)
            observed_label_values.update(str(value) for value in summary["unique_values"])

    print("\nSummary:")
    print(f"  total_files: {len(files)}")
    print(f"  approximate_images: {len(image_candidates)}")
    print(f"  approximate_masks: {len(mask_candidates)}")
    print(f"  geotiff_files: {len(rasters)}")
    print(f"  observed_shapes: {sorted(observed_shapes)}")
    print(f"  observed_label_values: {sorted(observed_label_values)}")
    if not files:
        print("  problems: dataset directory is empty")
    elif not rasters:
        print("  problems: no GeoTIFF files detected; download is likely incomplete")
    else:
        print("  problems: none detected")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local Sen1Floods11 directory.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/Sen1Floods11"))
    parser.add_argument("--max-rasters", type=int, default=5)
    parser.add_argument("--max-masks", type=int, default=5)
    args = parser.parse_args()
    inspect_dataset(args.root, args.max_rasters, args.max_masks)


if __name__ == "__main__":
    main()
