from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUCKET_HTTP = "https://copernicus-dem-30m.s3.amazonaws.com"
BUCKET_S3 = "s3://copernicus-dem-30m/"
TILELIST_URL = f"{BUCKET_HTTP}/tileList.txt"
RESOLUTION_TOKEN = "10"
DEFAULT_RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b2b_copernicus_dem_aws_download"
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_required_cells(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required cells CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ns_token(lat_floor: int) -> str:
    return f"{'N' if lat_floor >= 0 else 'S'}{abs(lat_floor):02d}_00"


def ew_token(lon_floor: int) -> str:
    return f"{'E' if lon_floor >= 0 else 'W'}{abs(lon_floor):03d}_00"


def tile_folder(lat_floor: int, lon_floor: int) -> str:
    return f"Copernicus_DSM_COG_{RESOLUTION_TOKEN}_{ns_token(lat_floor)}_{ew_token(lon_floor)}_DEM"


def tile_url(folder: str) -> str:
    return f"{BUCKET_HTTP}/{folder}/{folder}.tif"


def local_tile_path(output_root: Path, folder: str) -> Path:
    return output_root / f"{folder}.tif"


def download_url(url: str, output_path: Path, retries: int = 3, timeout: int = 120) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if partial_path.exists():
                partial_path.unlink()
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with partial_path.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            partial_path.replace(output_path)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if partial_path.exists():
                try:
                    partial_path.unlink()
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def fetch_tilelist(run_dir: Path) -> tuple[set[str], dict[str, Any]]:
    tilelist_path = run_dir / "metadata" / "tileList.txt"
    tilelist_path.parent.mkdir(parents=True, exist_ok=True)
    if not tilelist_path.exists():
        download_url(TILELIST_URL, tilelist_path, retries=3, timeout=120)
    folders = {
        line.strip().rstrip("/")
        for line in tilelist_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    summary = {
        "generated_at": now_utc(),
        "bucket": BUCKET_S3,
        "tilelist_url": TILELIST_URL,
        "tilelist_path": tilelist_path.as_posix(),
        "folder_count": len(folders),
        "resolution_token": RESOLUTION_TOKEN,
    }
    return folders, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def content_length(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def build_plan(
    *,
    required_cells: list[dict[str, str]],
    available_folders: set[str],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    planned: list[dict[str, Any]] = []
    available: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in required_cells:
        lat_floor = int(row["lat_floor"])
        lon_floor = int(row["lon_floor"])
        folder = tile_folder(lat_floor, lon_floor)
        url = tile_url(folder)
        target_path = local_tile_path(output_root, folder)
        exists_local = target_path.exists()
        item = {
            "cell_id": row["cell_id"],
            "lat_floor": lat_floor,
            "lon_floor": lon_floor,
            "north": int(row["north"]),
            "south": int(row["south"]),
            "east": int(row["east"]),
            "west": int(row["west"]),
            "sample_count": int(row["sample_count"]),
            "splits": row["splits"],
            "event_locations": row["event_locations"],
            "copernicus_folder": folder,
            "s3_bucket": BUCKET_S3,
            "url": url,
            "target_path": target_path.as_posix(),
            "exists_in_tilelist": folder in available_folders,
            "exists_local": exists_local,
            "local_size_bytes": content_length(target_path),
            "status": "available" if folder in available_folders else "missing_from_public_tilelist",
        }
        planned.append(item)
        if item["exists_in_tilelist"]:
            available.append(item)
        else:
            missing.append(item)
    return planned, available, missing


def download_one(row: dict[str, Any], retries: int) -> dict[str, Any]:
    target_path = Path(row["target_path"])
    if target_path.exists() and target_path.stat().st_size > 0:
        result = dict(row)
        result["download_status"] = "skipped_existing"
        result["downloaded_size_bytes"] = target_path.stat().st_size
        result["completed_at"] = now_utc()
        return result
    try:
        download_url(row["url"], target_path, retries=retries, timeout=180)
        result = dict(row)
        result["download_status"] = "downloaded"
        result["downloaded_size_bytes"] = target_path.stat().st_size
        result["completed_at"] = now_utc()
        return result
    except Exception as exc:  # noqa: BLE001
        result = dict(row)
        result["download_status"] = "failed"
        result["download_error"] = f"{type(exc).__name__}: {exc}"
        result["downloaded_size_bytes"] = content_length(target_path)
        result["completed_at"] = now_utc()
        return result


def summarize_required_cells(required_cells: list[dict[str, str]]) -> dict[str, Any]:
    lat_values = [int(row["lat_floor"]) for row in required_cells]
    lon_values = [int(row["lon_floor"]) for row in required_cells]
    event_locations = sorted(
        {
            event
            for row in required_cells
            for event in row.get("event_locations", "").split(";")
            if event
        }
    )
    splits = sorted(
        {split for row in required_cells for split in row.get("splits", "").split(";") if split}
    )
    return {
        "generated_at": now_utc(),
        "required_cell_count": len(required_cells),
        "event_locations": event_locations,
        "splits": splits,
        "min_lat_floor": min(lat_values),
        "max_lat_north": max(lat_values) + 1,
        "min_lon_floor": min(lon_values),
        "max_lon_east": max(lon_values) + 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download required Copernicus DEM GLO-30 AWS public tiles.")
    parser.add_argument("--required-cells", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source", choices=["copernicus_glo30"], default="copernicus_glo30")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--aws-no-sign-request", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["reports", "logs", "scripts", "manifests", "metadata", "downloads", "inventory"]:
        (args.run_dir / subdir).mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    required_cells = read_required_cells(args.required_cells)
    write_json(args.run_dir / "metadata" / "required_cells_summary.json", summarize_required_cells(required_cells))
    available_folders, tilelist_summary = fetch_tilelist(args.run_dir)
    planned, available, missing = build_plan(
        required_cells=required_cells,
        available_folders=available_folders,
        output_root=args.output_root,
    )

    fieldnames = list(planned[0].keys()) if planned else []
    write_csv(args.run_dir / "manifests" / "copernicus_required_tiles_planned.csv", planned, fieldnames)
    write_csv(args.run_dir / "manifests" / "copernicus_required_tiles_available.csv", available, fieldnames)
    write_csv(args.run_dir / "manifests" / "copernicus_required_tiles_missing.csv", missing, fieldnames)
    write_json(args.run_dir / "metadata" / "copernicus_tilelist_summary.json", tilelist_summary)

    downloaded: list[dict[str, Any]] = []
    download_started = False
    if args.download:
        download_started = True
        max_workers = max(1, int(args.max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_one, row, args.retries) for row in available]
            for future in as_completed(futures):
                result = future.result()
                downloaded.append(result)
                print(
                    f"{result['cell_id']} {result.get('download_status')} "
                    f"{result.get('downloaded_size_bytes', 0)} {result['target_path']}",
                    flush=True,
                )
    else:
        downloaded = [
            {
                **row,
                "download_status": "dry_run_only",
                "downloaded_size_bytes": row["local_size_bytes"],
                "completed_at": "",
            }
            for row in available
        ]

    downloaded_fieldnames = list(downloaded[0].keys()) if downloaded else fieldnames + ["download_status"]
    write_csv(args.run_dir / "manifests" / "copernicus_downloaded_tiles.csv", downloaded, downloaded_fieldnames)

    downloaded_count = sum(1 for row in downloaded if row.get("download_status") == "downloaded")
    skipped_count = sum(1 for row in downloaded if row.get("download_status") == "skipped_existing")
    failed_count = sum(1 for row in downloaded if row.get("download_status") == "failed")
    existing_local_count = sum(1 for row in planned if row["exists_local"])
    summary = {
        "step": "6B2b",
        "generated_at": now_utc(),
        "source": args.source,
        "bucket": BUCKET_S3,
        "bucket_http": BUCKET_HTTP,
        "aws_no_sign_request": bool(args.aws_no_sign_request),
        "required_cells": len(required_cells),
        "available_tiles": len(available),
        "missing_tiles": len(missing),
        "existing_local_tiles_before_download": existing_local_count,
        "download_requested": bool(args.download),
        "download_started": download_started,
        "downloaded_count": downloaded_count,
        "skipped_existing_count": skipped_count,
        "failed_count": failed_count,
        "download_completed": bool(args.download and failed_count == 0 and len(downloaded) == len(available)),
        "output_root": args.output_root.as_posix(),
        "planned_manifest": (args.run_dir / "manifests" / "copernicus_required_tiles_planned.csv").as_posix(),
        "available_manifest": (args.run_dir / "manifests" / "copernicus_required_tiles_available.csv").as_posix(),
        "missing_manifest": (args.run_dir / "manifests" / "copernicus_required_tiles_missing.csv").as_posix(),
        "downloaded_manifest": (args.run_dir / "manifests" / "copernicus_downloaded_tiles.csv").as_posix(),
    }
    write_json(args.run_dir / "metadata" / "copernicus_download_summary.json", summary)
    print(f"required_cells={len(required_cells)}")
    print(f"available_tiles={len(available)}")
    print(f"missing_tiles={len(missing)}")
    print(f"download_started={download_started}")
    print(f"download_completed={summary['download_completed']}")
    print(f"output_root={args.output_root}")
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
