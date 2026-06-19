from __future__ import annotations

import argparse
from pathlib import Path


def _stems(directory: Path) -> set[str]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return {path.stem for path in directory.iterdir() if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that image, mask and DEM files have matching filename stems."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--dems", type=Path, required=True)
    args = parser.parse_args()

    image_stems = _stems(args.images)
    mask_stems = _stems(args.masks)
    dem_stems = _stems(args.dems)

    common = image_stems & mask_stems & dem_stems
    print(f"images: {len(image_stems)}")
    print(f"masks: {len(mask_stems)}")
    print(f"dems: {len(dem_stems)}")
    print(f"aligned filename stems: {len(common)}")

    missing_masks = sorted(image_stems - mask_stems)
    missing_dems = sorted(image_stems - dem_stems)
    if missing_masks:
        print(f"images without mask: {missing_masks[:10]}")
    if missing_dems:
        print(f"images without DEM: {missing_dems[:10]}")

    # TODO: once dataset formats are known, validate geotransform, CRS, resolution,
    # raster shape and pixel-perfect alignment.


if __name__ == "__main__":
    main()
