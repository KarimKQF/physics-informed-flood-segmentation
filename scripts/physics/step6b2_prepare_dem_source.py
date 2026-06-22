from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b2_dem_source_acquisition"
)
DEFAULT_REQUIRED_CELLS = RUN_DIR / "manifests" / "required_dem_cells.csv"
SOURCE_INFO = {
    "copernicus_glo30": {
        "label": "Copernicus DEM GLO-30",
        "target_subdir": "copernicus_glo30",
        "extension": ".tif",
        "automatic_download_supported": False,
        "credential_hint": (
            "Use Copernicus Data Space registration or an explicitly configured public-data "
            "access method before enabling downloads."
        ),
    },
    "srtm_1arcsec": {
        "label": "SRTM Global 1 arc-second",
        "target_subdir": "srtm_1arcsec",
        "extension": ".hgt.zip",
        "automatic_download_supported": False,
        "credential_hint": (
            "Use NASA Earthdata/USGS EarthExplorer credentials or manually place required "
            "1-degree cells before alignment."
        ),
    },
}


def read_cells(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required DEM cells file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def planned_filename(source: str, cell_id: str) -> str:
    if source == "copernicus_glo30":
        return f"{cell_id}_COPERNICUS_GLO30.tif"
    if source == "srtm_1arcsec":
        return f"{cell_id}_SRTMGL1.hgt.zip"
    raise ValueError(f"Unsupported source: {source}")


def build_plan(source: str, output_root: Path, cells: list[dict[str, str]]) -> list[dict[str, Any]]:
    target_dir = output_root / SOURCE_INFO[source]["target_subdir"]
    rows: list[dict[str, Any]] = []
    for cell in cells:
        target_path = target_dir / planned_filename(source, cell["cell_id"])
        exists = target_path.exists()
        rows.append(
            {
                "source": source,
                "cell_id": cell["cell_id"],
                "lat_floor": int(cell["lat_floor"]),
                "lon_floor": int(cell["lon_floor"]),
                "north": int(cell["north"]),
                "south": int(cell["south"]),
                "east": int(cell["east"]),
                "west": int(cell["west"]),
                "sample_count": int(cell["sample_count"]),
                "splits": cell["splits"],
                "event_locations": cell["event_locations"],
                "target_path": target_path.as_posix(),
                "exists": exists,
                "status": "exists" if exists else "planned",
                "download_url": "",
                "notes": "URL construction/download intentionally not enabled in dry-run.",
            }
        )
    return rows


def write_plan(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "cell_id",
        "lat_floor",
        "lon_floor",
        "north",
        "south",
        "east",
        "west",
        "sample_count",
        "splits",
        "event_locations",
        "target_path",
        "exists",
        "status",
        "download_url",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or prepare required DEM source cells for Sen1Floods11."
    )
    parser.add_argument("--source", choices=sorted(SOURCE_INFO), default="copernicus_glo30")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--required-cells", type=Path, default=DEFAULT_REQUIRED_CELLS)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    try:
        cells = read_cells(args.required_cells)
        info = SOURCE_INFO[args.source]
        rows = build_plan(args.source, args.output_root, cells)
        csv_path = RUN_DIR / "manifests" / "planned_dem_download_manifest.csv"
        json_path = RUN_DIR / "manifests" / "planned_dem_download_manifest.json"
        write_plan(rows, csv_path, json_path)

        existing = sum(1 for row in rows if row["exists"])
        missing = len(rows) - existing
        result = {
            "step": "6B2",
            "source": args.source,
            "source_label": info["label"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "required_cell_count": len(rows),
            "existing_cell_count": existing,
            "missing_cell_count": missing,
            "output_root": args.output_root.as_posix(),
            "planned_manifest_csv": csv_path.as_posix(),
            "planned_manifest_json": json_path.as_posix(),
            "automatic_download_supported": info["automatic_download_supported"],
            "credential_hint": info["credential_hint"],
            "dry_run": not args.download,
            "download_requested": bool(args.download),
            "download_started": False,
            "download_completed": False,
            "environment": {
                "COPERNICUS_DEM_ACCESS_CONFIG": bool(os.environ.get("COPERNICUS_DEM_ACCESS_CONFIG")),
                "EARTHDATA_USERNAME": bool(os.environ.get("EARTHDATA_USERNAME")),
                "EARTHDATA_PASSWORD": bool(os.environ.get("EARTHDATA_PASSWORD")),
            },
        }
        (RUN_DIR / "metadata").mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "metadata" / "step6b2_dry_run_summary.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print(f"source={args.source}")
        print(f"source_label={info['label']}")
        print(f"required_cells={len(rows)}")
        print(f"existing_cells={existing}")
        print(f"missing_cells={missing}")
        print(f"planned_manifest_csv={csv_path}")
        print(f"planned_manifest_json={json_path}")
        print(f"automatic_download_supported={info['automatic_download_supported']}")
        print(f"credential_hint={info['credential_hint']}")

        if args.download:
            print(
                "[ERROR] Download mode was requested, but no supported authenticated access "
                "method is configured in this helper. Run dry-run and provide files/credentials manually.",
                file=sys.stderr,
            )
            return 2
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
