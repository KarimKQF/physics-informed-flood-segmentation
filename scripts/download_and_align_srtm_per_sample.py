from __future__ import annotations

import argparse
import csv
import gzip
import math
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject, transform_bounds

SRTM_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
MANIFEST_COLUMNS = ["sample_id", "image_path", "mask_path", "dem_path", "split"]
SRTM_VOID_VALUE = -32768.0


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


def tile_url(tile_name: str) -> str:
    return f"{SRTM_BASE_URL}/{tile_name[:3]}/{tile_name}.hgt.gz"


def read_manifest(manifest_path: Path, max_samples: int | None) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not set(MANIFEST_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns: {MANIFEST_COLUMNS}")
        rows = list(reader)
    return rows[:max_samples] if max_samples is not None else rows


def image_bounds_wgs84(image: rasterio.io.DatasetReader) -> tuple[float, float, float, float]:
    if image.crs is None:
        raise ValueError(f"Image has no CRS: {image.name}")
    bounds = image.bounds
    if str(image.crs) == "EPSG:4326":
        return bounds.left, bounds.bottom, bounds.right, bounds.top
    return transform_bounds(image.crs, "EPSG:4326", *bounds, densify_pts=21)


def sample_required_tiles(image_path: Path, buffer_degrees: float) -> list[str]:
    with rasterio.open(image_path) as image:
        left, bottom, right, top = image_bounds_wgs84(image)
    bounds = (
        left - buffer_degrees,
        bottom - buffer_degrees,
        right + buffer_degrees,
        top + buffer_degrees,
    )
    return srtm_tile_names(bounds)


def download_or_reuse_srtm_tile(tile_name: str, tile_cache_dir: Path) -> tuple[Path, bool, str]:
    tile_cache_dir.mkdir(parents=True, exist_ok=True)
    hgt_path = tile_cache_dir / f"{tile_name}.hgt"
    gz_path = tile_cache_dir / f"{tile_name}.hgt.gz"
    url = tile_url(tile_name)

    if hgt_path.exists():
        print(f"[INFO] Using cached tile: {hgt_path}")
        return hgt_path, False, url

    print(f"[INFO] Downloading missing tile: {url}")
    try:
        with urlopen(url, timeout=60) as response, gz_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    except HTTPError as exc:
        raise RuntimeError(f"Could not download SRTM tile {tile_name} from {url}: {exc}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach SRTM tile {tile_name} at {url}: {exc}") from exc

    try:
        with gzip.open(gz_path, "rb") as source, hgt_path.open("wb") as output:
            shutil.copyfileobj(source, output)
    except OSError as exc:
        raise RuntimeError(
            f"Could not decompress SRTM tile {tile_name} from {gz_path}: {exc}"
        ) from exc
    return hgt_path, True, url


def fill_aligned_dem(values: np.ndarray, sample_id: str) -> np.ndarray:
    values = np.where(values <= SRTM_VOID_VALUE + 1.0, np.nan, values)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"No finite DEM values after alignment for {sample_id}")
    fill_value = float(np.median(values[finite]))
    return np.where(finite, values, fill_value).astype("float32")


def align_tiles_to_reference(
    *,
    tile_paths: list[Path],
    image_path: Path,
    output_path: Path,
    sample_id: str,
) -> Path:
    if not tile_paths:
        raise ValueError(f"No SRTM tiles available for {sample_id}")

    with rasterio.open(image_path) as image:
        bounds_wgs84 = image_bounds_wgs84(image)
        destination = np.full((image.height, image.width), np.nan, dtype="float32")
        profile = image.profile.copy()
        dst_transform = image.transform
        dst_crs = image.crs
        dst_height = image.height
        dst_width = image.width

    sources = [rasterio.open(path) for path in tile_paths]
    try:
        source_nodata = sources[0].nodata
        if source_nodata is None:
            source_nodata = SRTM_VOID_VALUE
        mosaic, mosaic_transform = merge(sources, bounds=bounds_wgs84, nodata=source_nodata)
    finally:
        for source in sources:
            source.close()

    print("[INFO] Aligning DEM to reference grid")
    reproject(
        source=mosaic[0].astype("float32"),
        destination=destination,
        src_transform=mosaic_transform,
        src_crs="EPSG:4326",
        src_nodata=source_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    destination = fill_aligned_dem(destination, sample_id)

    profile.update(
        count=1,
        dtype="float32",
        height=dst_height,
        width=dst_width,
        nodata=None,
        compress="lzw",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(destination, 1)
    return output_path


def write_manifest(output_manifest: Path, rows: list[dict[str, str]]) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def download_and_align(
    *,
    manifest_path: Path,
    output_manifest: Path,
    tile_cache_dir: Path,
    aligned_dem_dir: Path,
    max_samples: int | None,
    max_tiles_total: int,
    overwrite: bool,
    buffer_degrees: float,
) -> dict[str, object]:
    if max_tiles_total <= 0:
        raise ValueError("--max-tiles-total must be positive.")

    rows = read_manifest(manifest_path, max_samples)
    if not rows:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")

    sample_tiles: dict[str, list[str]] = {}
    unique_tiles: set[str] = set()
    for row in rows:
        image_path = Path(row["image_path"])
        tiles = sample_required_tiles(image_path, buffer_degrees)
        sample_tiles[row["sample_id"]] = tiles
        unique_tiles.update(tiles)

    if len(unique_tiles) > max_tiles_total:
        raise RuntimeError(
            f"Selected samples require {len(unique_tiles)} unique SRTM tiles, "
            f"above --max-tiles-total={max_tiles_total}. Tiles: {sorted(unique_tiles)}"
        )

    downloaded_tiles: set[str] = set()
    cached_tiles: set[str] = set()
    updated_rows: list[dict[str, str]] = []
    created_dem_paths: list[str] = []

    for row in rows:
        sample_id = row["sample_id"]
        image_path = Path(row["image_path"])
        output_path = aligned_dem_dir / f"{sample_id}_DEM_aligned.tif"
        tiles = sample_tiles[sample_id]

        print(f"[INFO] Processing {sample_id}")
        print(f"[INFO] Required SRTM tiles: {tiles}")

        if output_path.exists() and not overwrite:
            print(f"[OK] DEM already aligned: {output_path}")
            updated_rows.append(dict(row, dem_path=output_path.as_posix()))
            created_dem_paths.append(output_path.as_posix())
            continue

        tile_paths = []
        for tile in tiles:
            tile_path, downloaded, _ = download_or_reuse_srtm_tile(tile, tile_cache_dir)
            tile_paths.append(tile_path)
            if downloaded:
                downloaded_tiles.add(tile)
            else:
                cached_tiles.add(tile)

        aligned_path = align_tiles_to_reference(
            tile_paths=tile_paths,
            image_path=image_path,
            output_path=output_path,
            sample_id=sample_id,
        )
        print(f"[OK] DEM aligned: {aligned_path}")
        updated_rows.append(dict(row, dem_path=aligned_path.as_posix()))
        created_dem_paths.append(aligned_path.as_posix())

    write_manifest(output_manifest, updated_rows)
    summary = {
        "samples": len(updated_rows),
        "unique_tiles": sorted(unique_tiles),
        "downloaded_tiles": sorted(downloaded_tiles),
        "cached_tiles": sorted(cached_tiles),
        "dem_files": created_dem_paths,
        "output_manifest": output_manifest.as_posix(),
    }
    print("[OK] DEM alignment completed")
    print(f"[INFO] DEM files created: {created_dem_paths}")
    print(f"[INFO] Downloaded tiles: {summary['downloaded_tiles']}")
    print(f"[INFO] Cached tiles reused: {summary['cached_tiles']}")
    print(f"[INFO] Output manifest: {output_manifest}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download cached SRTM tiles and align DEMs per Sen1Floods11 sample."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--tile-cache-dir", type=Path, required=True)
    parser.add_argument("--aligned-dem-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-tiles-total", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--buffer-degrees", type=float, default=0.0)
    args = parser.parse_args()

    try:
        download_and_align(
            manifest_path=args.manifest,
            output_manifest=args.output_manifest,
            tile_cache_dir=args.tile_cache_dir,
            aligned_dem_dir=args.aligned_dem_dir,
            max_samples=args.max_samples,
            max_tiles_total=args.max_tiles_total,
            overwrite=args.overwrite,
            buffer_degrees=args.buffer_degrees,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
