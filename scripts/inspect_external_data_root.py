from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.utils import ensure_external_data_tree, load_local_paths  # noqa: E402

RASTER_SUFFIXES = {".tif", ".tiff"}
MANIFEST_COLUMNS = ["sample_id", "image_path", "mask_path", "dem_path", "split"]


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def folder_stats(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    files = [item for item in path.rglob("*") if item.is_file()]
    geotiffs = [item for item in files if item.suffix.lower() in RASTER_SUFFIXES]
    size = sum(item.stat().st_size for item in files)
    return len(files), len(geotiffs), size


def ensure_manifest_template(manifests_dir: Path) -> Path:
    manifest_path = manifests_dir / "sen1floods11_subset_manifest.csv"
    if not manifest_path.exists():
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(MANIFEST_COLUMNS)
        print(f"[OK] manifest template created: {manifest_path}")
    else:
        print(f"[OK] manifest template exists: {manifest_path}")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and initialize external data root.")
    parser.add_argument("--config", type=Path, default=Path("configs/local_paths.yaml"))
    args = parser.parse_args()

    try:
        paths = load_local_paths(args.config)
        ensure_external_data_tree(paths)
        print(f"[OK] data_root exists: {paths.data_root}")

        for label, directory in paths.all_directories().items():
            files, geotiffs, size = folder_stats(directory)
            print(f"[OK] {label} exists: {directory}")
            print(f"[INFO] Number of files: {files}")
            print(f"[INFO] Number of GeoTIFF files: {geotiffs}")
            print(f"[INFO] Folder size: {format_bytes(size)}")

        ensure_manifest_template(paths.manifests)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
