from __future__ import annotations

import argparse
import csv
import gzip
import math
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.utils import ensure_external_data_tree, load_local_paths  # noqa: E402

SRTM_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def manifest_bounds_wgs84(
    rows: list[dict[str, str]],
    buffer_degrees: float,
) -> tuple[float, float, float, float]:
    if not rows:
        raise ValueError("Manifest contains no samples.")

    left = float("inf")
    bottom = float("inf")
    right = float("-inf")
    top = float("-inf")
    for row in rows:
        image_path = Path(row["image_path"])
        with rasterio.open(image_path) as image:
            bounds = image.bounds
            if image.crs is None:
                raise ValueError(f"Image has no CRS: {image_path}")
            if str(image.crs) != "EPSG:4326":
                bounds_tuple = transform_bounds(image.crs, "EPSG:4326", *bounds, densify_pts=21)
                sample_left, sample_bottom, sample_right, sample_top = bounds_tuple
            else:
                sample_left, sample_bottom, sample_right, sample_top = bounds
            left = min(left, sample_left)
            bottom = min(bottom, sample_bottom)
            right = max(right, sample_right)
            top = max(top, sample_top)

    return (
        left - buffer_degrees,
        bottom - buffer_degrees,
        right + buffer_degrees,
        top + buffer_degrees,
    )


def lat_label(lat: int) -> str:
    return f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"


def lon_label(lon: int) -> str:
    return f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"


def srtm_tile_names(bounds: tuple[float, float, float, float]) -> list[str]:
    left, bottom, right, top = bounds
    lon_values = range(math.floor(left), math.ceil(right))
    lat_values = range(math.floor(bottom), math.ceil(top))
    names = []
    for lat in lat_values:
        for lon in lon_values:
            names.append(f"{lat_label(lat)}{lon_label(lon)}")
    return names


def download_srtm_tile(tile_name: str, tile_dir: Path, overwrite: bool) -> Path:
    folder = tile_name[:3]
    gz_path = tile_dir / f"{tile_name}.hgt.gz"
    hgt_path = tile_dir / f"{tile_name}.hgt"
    if hgt_path.exists() and not overwrite:
        return hgt_path

    tile_dir.mkdir(parents=True, exist_ok=True)
    url = f"{SRTM_BASE_URL}/{folder}/{tile_name}.hgt.gz"
    print(f"[INFO] Downloading SRTM tile: {url}")
    try:
        with urlopen(url, timeout=60) as response, gz_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception as exc:
        raise RuntimeError(f"Could not download SRTM tile {tile_name} from {url}: {exc}") from exc

    with gzip.open(gz_path, "rb") as source, hgt_path.open("wb") as output:
        shutil.copyfileobj(source, output)
    return hgt_path


def create_raw_dem_from_srtm(
    *,
    manifest_path: Path,
    output_path: Path,
    tmp_dir: Path,
    overwrite: bool,
    buffer_degrees: float,
    max_tiles: int,
) -> Path:
    rows = read_manifest(manifest_path)
    bounds = manifest_bounds_wgs84(rows, buffer_degrees)
    tile_names = srtm_tile_names(bounds)
    if not tile_names:
        raise ValueError(f"No SRTM tiles selected for bounds: {bounds}")

    print(f"[INFO] WGS84 bounds with buffer: {bounds}")
    print(f"[INFO] SRTM tile count selected from global bounds: {len(tile_names)}")
    if len(tile_names) <= 20:
        print(f"[INFO] SRTM tiles selected: {tile_names}")
    else:
        preview = tile_names[:5] + ["..."] + tile_names[-5:]
        print(f"[INFO] SRTM tile preview: {preview}")
    if len(tile_names) > max_tiles:
        raise RuntimeError(
            f"SRTM selection covers {len(tile_names)} tiles, which is above the safety limit "
            f"of {max_tiles}. The subset may be geographically dispersed. Reduce the subset, "
            "increase --max-tiles deliberately, or provide a manually prepared DEM for this area."
        )

    if output_path.exists() and not overwrite:
        print(f"[OK] DEM already exists: {output_path}")
        return output_path

    tile_paths = [download_srtm_tile(name, tmp_dir, overwrite) for name in tile_names]
    sources = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, transform = merge(sources, bounds=bounds)
        nodata = sources[0].nodata
    finally:
        for source in sources:
            source.close()

    dem = mosaic[0].astype("float32")
    if nodata is not None:
        dem = np.where(dem == nodata, -9999.0, dem)
    dem = np.nan_to_num(dem, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
        "compress": "lzw",
    }
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(dem, 1)

    print(f"[OK] Raw DEM written to: {output_path}")
    print(f"[INFO] Raw DEM shape: {dem.shape}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a DEM covering a Sen1Floods11 subset.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/local_paths.yaml"))
    parser.add_argument("--source", choices=["srtm", "copernicus"], default="srtm")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--buffer-degrees", type=float, default=0.01)
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=16,
        help="Safety limit to avoid accidental large DEM downloads.",
    )
    args = parser.parse_args()

    try:
        paths = load_local_paths(args.config)
        ensure_external_data_tree(paths)
        if args.source != "srtm":
            raise RuntimeError(
                "Automatic Copernicus DEM download is not configured. "
                "Use --source srtm or provide a DEM manually in D:/urban_runoff_data/raw/DEM."
            )
        create_raw_dem_from_srtm(
            manifest_path=args.manifest,
            output_path=paths.raw_dem / "sen1floods11_subset_dem_raw.tif",
            tmp_dir=paths.tmp / "srtm_tiles",
            overwrite=args.overwrite,
            buffer_degrees=args.buffer_degrees,
            max_tiles=args.max_tiles,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
