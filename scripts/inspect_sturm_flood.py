from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

RASTER_SUFFIXES = (".tif", ".tiff")
SENTINEL1_HINTS = ("s1", "sentinel-1", "sentinel1", "sar", "vv", "vh")
SENTINEL2_HINTS = ("s2", "sentinel-2", "sentinel2", "rgb", "nir", "swir")
MASK_HINTS = ("mask", "label", "water", "flood", "inundation")


def _load_rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise SystemExit(
            "rasterio is required to inspect GeoTIFF metadata. "
            'Install dependencies with: pip install -e ".[dev]"'
        ) from exc
    return rasterio


def _contains_any(path: Path, hints: tuple[str, ...], root: Path | None = None) -> bool:
    candidate = path.relative_to(root) if root is not None else path
    lowered = str(candidate).lower()
    return any(hint in lowered for hint in hints)


def _list_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _print_examples(title: str, paths: list[Path], root: Path, limit: int = 10) -> None:
    print(f"{title}: {len(paths)}")
    for path in paths[:limit]:
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
            f"{root} does not exist. Run: python scripts/download_sturm_flood.py --dry-run"
        )

    files = _list_files(root)
    suffix_counts = Counter(path.suffix.lower() or "<no suffix>" for path in files)
    rasters = [path for path in files if path.suffix.lower() in RASTER_SUFFIXES]
    s1_candidates = [path for path in files if _contains_any(path, SENTINEL1_HINTS, root)]
    s2_candidates = [path for path in files if _contains_any(path, SENTINEL2_HINTS, root)]
    mask_candidates = [path for path in files if _contains_any(path, MASK_HINTS, root)]
    observed_shapes: set[tuple[int, int]] = set()
    observed_label_values: set[str] = set()

    print(f"Root: {root}")
    print(f"Files: {len(files)}")
    print(f"Suffix counts: {dict(suffix_counts.most_common(20))}")
    _print_examples("Example files", files, root)
    _print_examples("Sentinel-1 candidates", s1_candidates, root)
    _print_examples("Sentinel-2 candidates", s2_candidates, root)
    _print_examples("Mask candidates", mask_candidates, root)
    _print_examples("GeoTIFF candidates", rasters, root)

    print("\nRaster metadata samples:")
    for path in rasters[:max_rasters]:
        summary = _read_raster_summary(
            path,
            root,
            show_unique=_contains_any(path, MASK_HINTS, root),
        )
        shape = summary["shape"]
        if isinstance(shape, tuple):
            observed_shapes.add(shape)

    if mask_candidates:
        print("\nMask candidate value samples:")
        for path in [path for path in mask_candidates if path in rasters][:max_masks]:
            summary = _read_raster_summary(path, root, show_unique=True)
            observed_label_values.update(str(value) for value in summary["unique_values"])

    print("\nSummary:")
    print(f"  total_files: {len(files)}")
    print(f"  detected_formats: {dict(suffix_counts.most_common(20))}")
    print(f"  has_sentinel_1: {bool(s1_candidates)}")
    print(f"  has_sentinel_2: {bool(s2_candidates)}")
    print(f"  has_masks: {bool(mask_candidates)}")
    print(f"  geotiff_files: {len(rasters)}")
    print(f"  observed_shapes: {sorted(observed_shapes)}")
    print(f"  observed_label_values: {sorted(observed_label_values)}")
    print("  problems: none detected" if files else "  problems: dataset directory is empty")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local STURM-Flood directory.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/STURM-Flood"))
    parser.add_argument("--max-rasters", type=int, default=5)
    parser.add_argument("--max-masks", type=int, default=5)
    args = parser.parse_args()
    inspect_dataset(args.root, args.max_rasters, args.max_masks)


if __name__ == "__main__":
    main()
