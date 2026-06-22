from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds


RASTER_EXTENSIONS = {".tif", ".tiff", ".vrt", ".img", ".bil", ".dem"}
MANIFEST_FIELDS = [
    "tile_id",
    "split",
    "event_location",
    "s1_path",
    "s2_path",
    "label_path",
    "topography_path",
    "topography_type",
    "crs",
    "height",
    "width",
    "finite_ratio",
    "nodata_ratio",
    "status",
    "notes",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_sources(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in RASTER_EXTENSIONS)
    raise FileNotFoundError(f"DEM source path does not exist: {source}")


def bounds_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def choose_source_for_label(sources: list[Path], label) -> Path | None:
    label_bounds = (label.bounds.left, label.bounds.bottom, label.bounds.right, label.bounds.top)
    for source in sources:
        try:
            with rasterio.open(source) as src:
                if src.crs is None:
                    continue
                src_bounds = (
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top,
                )
                if src.crs != label.crs:
                    src_bounds = transform_bounds(src.crs, label.crs, *src_bounds, densify_pts=21)
                if bounds_intersect(src_bounds, label_bounds):
                    return source
        except Exception:
            continue
    return None


def finite_stats(array: np.ndarray, nodata: float | None) -> dict[str, float | None]:
    finite_mask = np.isfinite(array)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        finite_mask &= array != nodata
    finite = array[finite_mask]
    total = array.size
    if finite.size == 0:
        return {
            "finite_ratio": 0.0,
            "nodata_ratio": 1.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "finite_ratio": float(finite.size / total),
        "nodata_ratio": float(1.0 - finite.size / total),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def align_one(
    *,
    row: dict[str, str],
    source_path: Path,
    output_dir: Path,
    topography_type: str,
    output_format: str,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tile_id = row["tile_id"]
    label_path = Path(row["label_path"])
    suffix = ".tif" if output_format == "geotiff" else ".npz"
    output_path = output_dir / f"{tile_id}_{topography_type}_aligned{suffix}"
    if output_path.exists() and not overwrite:
        return (
            {
                "tile_id": tile_id,
                "split": row.get("split", ""),
                "event_location": row.get("event_location", ""),
                "s1_path": row.get("s1_path", ""),
                "s2_path": row.get("s2_path", ""),
                "label_path": row.get("label_path", ""),
                "topography_path": output_path.as_posix(),
                "topography_type": topography_type,
                "crs": "",
                "height": "",
                "width": "",
                "finite_ratio": "",
                "nodata_ratio": "",
                "status": "exists_skipped",
                "notes": "Existing output kept because --overwrite was not set.",
            },
            {"tile_id": tile_id, "status": "exists_skipped"},
        )

    with rasterio.open(label_path) as label, rasterio.open(source_path) as src:
        destination = np.full((label.height, label.width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=label.transform,
            dst_crs=label.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        stats = finite_stats(destination, np.nan)
        if output_format == "geotiff":
            profile = label.profile.copy()
            profile.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(destination, 1)
            with rasterio.open(output_path) as aligned:
                shape_ok = aligned.height == label.height and aligned.width == label.width
                crs_ok = aligned.crs == label.crs
                transform_ok = tuple(aligned.transform) == tuple(label.transform)
        else:
            np.savez_compressed(
                output_path,
                topography=destination,
                transform=np.asarray(label.transform),
                crs=str(label.crs),
                tile_id=tile_id,
            )
            shape_ok = destination.shape == (label.height, label.width)
            crs_ok = True
            transform_ok = True

        status = "ok" if shape_ok and crs_ok and transform_ok and stats["finite_ratio"] > 0 else "failed_qc"
        manifest_row = {
            "tile_id": tile_id,
            "split": row.get("split", ""),
            "event_location": row.get("event_location", ""),
            "s1_path": row.get("s1_path", ""),
            "s2_path": row.get("s2_path", ""),
            "label_path": row.get("label_path", ""),
            "topography_path": output_path.as_posix(),
            "topography_type": topography_type,
            "crs": str(label.crs),
            "height": label.height,
            "width": label.width,
            "finite_ratio": stats["finite_ratio"],
            "nodata_ratio": stats["nodata_ratio"],
            "status": status,
            "notes": f"source={source_path.as_posix()}",
        }
        qc = {
            "tile_id": tile_id,
            "status": status,
            "output_path": output_path.as_posix(),
            "source_path": source_path.as_posix(),
            "shape_ok": shape_ok,
            "crs_ok": crs_ok,
            "transform_ok": transform_ok,
            **stats,
        }
        return manifest_row, qc


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align a DEM/topographic raster source to Sen1Floods11 LabelHand grids."
    )
    parser.add_argument("--source-dem", type=Path, required=True)
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest-csv", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--qc-json", type=Path, required=True)
    parser.add_argument("--topography-type", default="dem")
    parser.add_argument("--output-format", choices=["geotiff", "npz"], default="geotiff")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--tile-ids", nargs="*", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        sources = collect_sources(args.source_dem)
        if not sources:
            raise FileNotFoundError(f"No raster sources found under {args.source_dem}")
        rows = read_rows(args.inventory_csv)
        rows = [row for row in rows if row.get("status") == "ok"]
        if args.splits:
            allowed = set(args.splits)
            rows = [row for row in rows if row.get("split") in allowed]
        if args.tile_ids:
            wanted = set(args.tile_ids)
            rows = [row for row in rows if row.get("tile_id") in wanted]
        if args.max_tiles is not None:
            rows = rows[: args.max_tiles]
        if not rows:
            raise ValueError("No inventory rows selected for alignment.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, Any]] = []
        qc_rows: list[dict[str, Any]] = []
        for row in rows:
            with rasterio.open(row["label_path"]) as label:
                source = choose_source_for_label(sources, label)
            if source is None:
                qc_rows.append({"tile_id": row["tile_id"], "status": "no_intersecting_source"})
                continue
            manifest_row, qc = align_one(
                row=row,
                source_path=source,
                output_dir=args.output_dir,
                topography_type=args.topography_type,
                output_format=args.output_format,
                overwrite=args.overwrite,
            )
            manifest_rows.append(manifest_row)
            qc_rows.append(qc)

        write_manifest(args.output_manifest_csv, manifest_rows)
        args.output_manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest_json.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
        args.qc_json.parent.mkdir(parents=True, exist_ok=True)
        args.qc_json.write_text(
            json.dumps(
                {
                    "source_dem": args.source_dem.as_posix(),
                    "inventory_csv": args.inventory_csv.as_posix(),
                    "output_dir": args.output_dir.as_posix(),
                    "selected_rows": len(rows),
                    "written_rows": len(manifest_rows),
                    "qc": qc_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"written_rows={len(manifest_rows)}")
        print(f"manifest={args.output_manifest_csv}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
