from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_ok(message: str) -> None:
    print(f"[OK] {message}")


def ensure_input_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def raster_extent(bounds) -> list[float]:
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def visualize_aligned_building_mask(
    reference_path: Path,
    mask_path: Path,
    output_path: Path,
    buildings_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    reference_path = Path(reference_path)
    mask_path = Path(mask_path)
    output_path = Path(output_path)
    buildings_path = Path(buildings_path) if buildings_path else None

    ensure_input_file(reference_path, "Reference raster")
    ensure_input_file(mask_path, "Building mask")
    if buildings_path:
        ensure_input_file(buildings_path, "Buildings GeoJSON")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    log_info(f"Loading reference raster: {reference_path}")
    with rasterio.open(reference_path) as reference:
        reference_data = reference.read(1)
        reference_crs = reference.crs
        reference_bounds = reference.bounds
        reference_shape = (reference.height, reference.width)

    log_info(f"Loading building mask: {mask_path}")
    with rasterio.open(mask_path) as mask:
        mask_data = mask.read(1)
        mask_crs = mask.crs
        mask_shape = (mask.height, mask.width)

    if reference_shape != mask_shape:
        raise ValueError(
            "Reference raster and mask shapes differ: "
            f"reference={reference_shape}, mask={mask_shape}"
        )
    if reference_crs != mask_crs:
        raise ValueError(f"Reference raster and mask CRS differ: {reference_crs} vs {mask_crs}")

    extent = raster_extent(reference_bounds)
    masked_buildings = np.ma.masked_where(mask_data == 0, mask_data)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(reference_data, extent=extent, cmap="Greys", alpha=0.18, origin="upper")
    ax.imshow(masked_buildings, extent=extent, cmap="viridis", alpha=0.78, origin="upper")

    if buildings_path:
        log_info(f"Loading building outlines: {buildings_path}")
        buildings = gpd.read_file(buildings_path)
        if buildings.empty:
            log_info("Buildings GeoJSON is empty; skipping outlines.")
        else:
            if buildings.crs is None:
                raise ValueError(f"Buildings GeoJSON has no CRS: {buildings_path}")
            if buildings.crs != reference_crs:
                log_info("Reprojecting building outlines to reference CRS...")
                buildings = buildings.to_crs(reference_crs)
            buildings.boundary.plot(ax=ax, color="#111111", linewidth=0.25, alpha=0.45)

    ax.set_title("Aligned OSM building mask on development reference raster")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Writing figure to: {output_path}")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    log_ok("Aligned building mask visualization successfully created.")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize an aligned building mask on its reference raster grid."
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
    parser.add_argument(
        "--buildings",
        type=Path,
        default=None,
        help="Optional path to the OSM buildings GeoJSON for outline overlay.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the PNG figure will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output PNG if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        visualize_aligned_building_mask(
            reference_path=args.reference,
            mask_path=args.mask,
            buildings_path=args.buildings,
            output_path=args.output,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
