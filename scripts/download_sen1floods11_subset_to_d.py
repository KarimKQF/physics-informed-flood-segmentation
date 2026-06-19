from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.utils import ensure_external_data_tree, load_local_paths  # noqa: E402

RASTER_SUFFIXES = (".tif", ".tiff")
MASK_KEYWORDS = ("label", "mask", "water", "labelhand", "s1otsulabel", "jrcwater")
MANIFEST_COLUMNS = ["sample_id", "image_path", "mask_path", "dem_path", "split"]
IMAGE_SUFFIXES = ("_S1Hand", "_S1Weak")
MASK_SUFFIXES = ("_LabelHand", "_LabelWeak")


def find_gsutil() -> str | None:
    candidates = [
        shutil.which("gsutil"),
        shutil.which("gsutil.cmd"),
        r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd",
        r"C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def is_mask_like(uri: str) -> bool:
    lowered = uri.rsplit("/", maxsplit=1)[-1].lower()
    return any(keyword in lowered for keyword in MASK_KEYWORDS)


def source_sort_key(uri: str) -> tuple[int, str]:
    lowered = uri.lower()
    priority = 2
    if "handlabeled" in lowered and "s1hand" in lowered:
        priority = 0
    elif "handlabeled" in lowered and "labelhand" in lowered:
        priority = 1
    return priority, uri


def sample_key(path_or_uri: str) -> str:
    stem = Path(path_or_uri.rsplit("/", maxsplit=1)[-1]).stem
    for suffix in (*IMAGE_SUFFIXES, *MASK_SUFFIXES, "_JRCWaterHand", "_S1OtsuLabelHand"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def is_primary_image(uri: str) -> bool:
    stem = Path(uri.rsplit("/", maxsplit=1)[-1]).stem
    return stem.endswith(IMAGE_SUFFIXES)


def is_primary_mask(uri: str) -> bool:
    stem = Path(uri.rsplit("/", maxsplit=1)[-1]).stem
    return stem.endswith(MASK_SUFFIXES)


def select_subset(geotiffs: list[str], max_files: int) -> list[str]:
    images: dict[str, str] = {}
    masks: dict[str, str] = {}
    for uri in sorted(geotiffs):
        key = sample_key(uri)
        if is_primary_image(uri):
            images.setdefault(key, uri)
        elif is_primary_mask(uri):
            masks.setdefault(key, uri)

    selected: list[str] = []
    for key in sorted(set(images) & set(masks)):
        if len(selected) + 2 > max_files:
            break
        selected.extend([images[key], masks[key]])

    if selected:
        return selected
    return sorted(geotiffs, key=source_sort_key)[:max_files]


def list_remote_geotiffs(gsutil: str, timeout: float) -> list[str]:
    command = [gsutil, "ls", "-r", "gs://sen1floods11/v1.1/data/**"]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not list remote Sen1Floods11 GeoTIFFs.\n"
            f"Command: {' '.join(command)}\n{completed.stderr}"
        )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().lower().endswith(RASTER_SUFFIXES)
    ]


def destination_for_uri(uri: str, output_root: Path) -> Path:
    marker = "gs://sen1floods11/"
    if uri.startswith(marker):
        relative = uri.removeprefix(marker)
    else:
        relative = Path(uri).name
    return output_root / relative


def copy_uri(gsutil: str, uri: str, destination: Path, overwrite: bool, timeout: float) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return "exists"
    command = [gsutil, "cp", uri, str(destination)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"Command failed: {' '.join(command)}")
    return "downloaded"


def create_best_effort_manifest(root: Path, manifest_path: Path) -> int:
    rasters = sorted(path for path in root.rglob("*") if path.suffix.lower() in RASTER_SUFFIXES)
    images = {sample_key(str(path)): path for path in rasters if is_primary_image(str(path))}
    masks = {sample_key(str(path)): path for path in rasters if is_primary_mask(str(path))}
    rows: list[dict[str, str]] = []

    for index, key in enumerate(sorted(set(images) & set(masks)), start=1):
        image_path = images[key]
        mask_path = masks[key]
        rows.append(
            {
                "sample_id": f"sample_{index:03d}",
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "dem_path": "",
                "split": "train" if index > 1 else "val",
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a small Sen1Floods11 subset to D:.")
    parser.add_argument("--config", type=Path, default=Path("configs/local_paths.yaml"))
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--include-tif-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    try:
        paths = load_local_paths(args.config)
        ensure_external_data_tree(paths)
        gsutil = find_gsutil()
        log_path = paths.logs / "sen1floods11_subset_download.log"
        manifest_path = paths.manifests / "sen1floods11_subset_manifest.csv"

        with log_path.open("w", encoding="utf-8") as log:
            if gsutil is None:
                message = "gsutil was not found. Install Google Cloud SDK and retry."
                log.write(f"[ERROR] {message}\n")
                print(f"[ERROR] {message}")
                return 1

            geotiffs = list_remote_geotiffs(gsutil, args.timeout)
            selected = select_subset(geotiffs, args.max_files)
            log.write(f"Remote GeoTIFF candidates: {len(geotiffs)}\n")
            log.write(f"Selected files: {len(selected)}\n")

            downloaded = 0
            failed = 0
            for uri in selected:
                destination = destination_for_uri(uri, paths.raw_sen1floods11)
                log.write(f"{uri} -> {destination}\n")
                if args.dry_run:
                    continue
                try:
                    status = copy_uri(gsutil, uri, destination, args.overwrite, args.timeout)
                    downloaded += int(status == "downloaded")
                    log.write(f"  status: {status}\n")
                except Exception as exc:
                    failed += 1
                    log.write(f"  failed: {exc}\n")

            row_count = create_best_effort_manifest(paths.raw_sen1floods11, manifest_path)
            log.write(f"Manifest rows: {row_count}\n")
            log.write(f"Failed downloads: {failed}\n")

        print(f"[OK] Log written to: {log_path}")
        print(f"[INFO] Selected remote files: {len(selected)}")
        print(f"[INFO] Downloaded files: {downloaded if not args.dry_run else 0}")
        print(f"[INFO] Manifest written to: {manifest_path}")
        print(f"[INFO] Manifest rows: {row_count}")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
