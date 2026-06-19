from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


def log_ok(message: str) -> None:
    print(f"[OK] {message}")


def log_error(message: str) -> None:
    print(f"[ERROR] {message}")


def ensure_input_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def transform_values(transform) -> tuple[float, float, float, float, float, float]:
    return tuple(transform)[:6]


def bounds_values(bounds) -> tuple[float, float, float, float]:
    return (bounds.left, bounds.bottom, bounds.right, bounds.top)


def validate_alignment(reference_path: Path, mask_path: Path) -> bool:
    reference_path = Path(reference_path)
    mask_path = Path(mask_path)
    ensure_input_file(reference_path, "Reference raster")
    ensure_input_file(mask_path, "Building mask")

    passed = True

    with rasterio.open(reference_path) as reference, rasterio.open(mask_path) as mask:
        if reference.crs == mask.crs:
            log_ok("CRS aligned")
        else:
            log_error("CRS mismatch")
            print(f"Reference CRS: {reference.crs}")
            print(f"Mask CRS: {mask.crs}")
            passed = False

        if np.allclose(transform_values(reference.transform), transform_values(mask.transform)):
            log_ok("transform aligned")
        else:
            log_error("Transform mismatch")
            print(f"Reference transform: {reference.transform}")
            print(f"Mask transform: {mask.transform}")
            passed = False

        if reference.width == mask.width and reference.height == mask.height:
            log_ok("shape aligned")
        else:
            log_error("Shape mismatch")
            print(f"Reference shape: height={reference.height}, width={reference.width}")
            print(f"Mask shape: height={mask.height}, width={mask.width}")
            passed = False

        if np.allclose(bounds_values(reference.bounds), bounds_values(mask.bounds)):
            log_ok("bounds aligned")
        else:
            log_error("Bounds mismatch")
            print(f"Reference bounds: {reference.bounds}")
            print(f"Mask bounds: {mask.bounds}")
            passed = False

        if np.allclose(reference.res, mask.res):
            log_ok("resolution aligned")
        else:
            log_error("Resolution mismatch")
            print(f"Reference resolution: {reference.res}")
            print(f"Mask resolution: {mask.res}")
            passed = False

        if reference.indexes and mask.indexes:
            log_ok("raster dimensions readable")
        else:
            log_error("Raster dimensions are not readable")
            passed = False

        if mask.count == 1:
            log_ok("mask is single-band")
        else:
            log_error("Mask band count mismatch")
            print(f"Mask bands: {mask.count}")
            passed = False

        mask_data = mask.read(1)
        unique_values = np.unique(mask_data)
        if set(unique_values.tolist()).issubset({0, 1}):
            log_ok("mask is binary")
        else:
            log_error("Mask has non-binary values")
            print(f"Mask unique values: {unique_values.tolist()}")
            passed = False

        if mask.dtypes[0] == "uint8":
            log_ok("dtype is uint8")
        else:
            log_error("Mask dtype is not uint8")
            print(f"Mask dtype: {mask.dtypes[0]}")
            passed = False

    if passed:
        log_ok("Alignment validation passed.")
    else:
        print("[FAILED] Alignment validation failed.")
    return passed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that a building mask is aligned with a reference raster."
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Path to the reference GeoTIFF raster.",
    )
    parser.add_argument(
        "--mask",
        required=True,
        type=Path,
        help="Path to the aligned building mask GeoTIFF.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return 0 if validate_alignment(args.reference, args.mask) else 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
