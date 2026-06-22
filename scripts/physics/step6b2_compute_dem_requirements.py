from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b2_dem_source_acquisition"
)
STEP6B_RUN = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b_topographic_alignment_validation"
)
GEOSPATIAL_INVENTORY_CSV = STEP6B_RUN / "inventory" / "sen1floods11_geospatial_inventory.csv"
STEP6B_DEM_AVAILABILITY = STEP6B_RUN / "inventory" / "dem_source_availability.json"
DATA_ROOT = Path("E:/flood_research/data")

RASTER_EXTENSIONS = {".tif", ".tiff", ".vrt", ".img", ".hgt", ".dem"}
DEM_KEYWORDS = [
    "copernicus",
    "glo30",
    "glo-30",
    "srtm",
    "nasadem",
    "elevation",
    "dem",
    "hand",
    "dtm",
    "dsm",
]
FALSE_HAND_MARKERS = [
    "handlabeled",
    "labelhand",
    "s1hand",
    "s2hand",
    "jrcwaterhand",
    "s1otsulabelhand",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cell_id(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"


def parse_bounds(row: dict[str, str]) -> tuple[float, float, float, float]:
    bounds = json.loads(row["label_bounds"])
    west, south, east, north = [float(value) for value in bounds]
    return west, south, east, north


def cells_for_bounds(west: float, south: float, east: float, north: float) -> list[tuple[int, int]]:
    lon_start = math.floor(west)
    lon_end = math.ceil(east)
    lat_start = math.floor(south)
    lat_end = math.ceil(north)
    cells: list[tuple[int, int]] = []
    for lat in range(lat_start, lat_end):
        for lon in range(lon_start, lon_end):
            cells.append((lat, lon))
    return cells


def is_false_hand_path(path: Path) -> bool:
    lowered = path.as_posix().lower()
    return any(marker in lowered for marker in FALSE_HAND_MARKERS)


def rescan_dem_sources() -> dict[str, Any]:
    candidates: list[str] = []
    scanned_file_count = 0
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file():
            continue
        scanned_file_count += 1
        lowered = path.as_posix().lower()
        if path.suffix.lower() not in RASTER_EXTENSIONS:
            continue
        if not any(keyword in lowered for keyword in DEM_KEYWORDS):
            continue
        if "hand" in lowered and is_false_hand_path(path):
            continue
        candidates.append(path.as_posix())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root_scanned": DATA_ROOT.as_posix(),
        "scanned_file_count": scanned_file_count,
        "raster_extensions": sorted(RASTER_EXTENSIONS),
        "keywords": DEM_KEYWORDS,
        "candidate_raster_count": len(candidates),
        "candidate_rasters": candidates,
        "dem_source_available": len(candidates) > 0,
        "note": (
            "HandLabeled/LabelHand/S1Hand/S2Hand files are intentionally excluded because "
            "they are not HAND topography."
        ),
    }


def write_verified_empty(post_scan: dict[str, Any]) -> None:
    csv_path = RUN_DIR / "inventory" / "verified_dem_source_inventory.csv"
    json_path = RUN_DIR / "inventory" / "verified_dem_source_inventory.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "source_type",
        "exists",
        "verified",
        "crs",
        "bounds",
        "nodata",
        "resolution",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
    write_json(
        json_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "verified_count": 0,
            "dem_source_available": bool(post_scan.get("dem_source_available", False)),
            "reason": "No candidate DEM raster files found for verification.",
            "rows": [],
        },
    )


def main() -> int:
    for subdir in ["reports", "logs", "inventory", "scripts", "manifests", "downloads", "metadata"]:
        (RUN_DIR / subdir).mkdir(parents=True, exist_ok=True)

    rows = read_rows(GEOSPATIAL_INVENTORY_CSV)
    if not rows:
        raise RuntimeError(f"No rows found in {GEOSPATIAL_INVENTORY_CSV}")

    total_west = math.inf
    total_south = math.inf
    total_east = -math.inf
    total_north = -math.inf
    event_locations = sorted({row["event_location"] for row in rows})
    crs_values = sorted({row["label_crs"] for row in rows})
    cell_payload: dict[tuple[int, int], dict[str, Any]] = {}
    per_split: dict[str, set[str]] = defaultdict(set)
    per_event: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        west, south, east, north = parse_bounds(row)
        total_west = min(total_west, west)
        total_south = min(total_south, south)
        total_east = max(total_east, east)
        total_north = max(total_north, north)
        for lat_floor, lon_floor in cells_for_bounds(west, south, east, north):
            key = (lat_floor, lon_floor)
            cid = cell_id(lat_floor, lon_floor)
            if key not in cell_payload:
                cell_payload[key] = {
                    "cell_id": cid,
                    "lat_floor": lat_floor,
                    "lon_floor": lon_floor,
                    "north": lat_floor + 1,
                    "south": lat_floor,
                    "east": lon_floor + 1,
                    "west": lon_floor,
                    "tile_ids": set(),
                    "splits": set(),
                    "event_locations": set(),
                }
            cell_payload[key]["tile_ids"].add(row["tile_id"])
            cell_payload[key]["splits"].add(row["split"])
            cell_payload[key]["event_locations"].add(row["event_location"])
            per_split[row["split"]].add(cid)
            per_event[row["event_location"]].add(cid)

    cells = []
    for payload in sorted(cell_payload.values(), key=lambda item: item["cell_id"]):
        cells.append(
            {
                "cell_id": payload["cell_id"],
                "lat_floor": payload["lat_floor"],
                "lon_floor": payload["lon_floor"],
                "north": payload["north"],
                "south": payload["south"],
                "east": payload["east"],
                "west": payload["west"],
                "sample_count": len(payload["tile_ids"]),
                "splits": ";".join(sorted(payload["splits"])),
                "event_locations": ";".join(sorted(payload["event_locations"])),
            }
        )

    cells_csv = RUN_DIR / "manifests" / "required_dem_cells.csv"
    cells_json = RUN_DIR / "manifests" / "required_dem_cells.json"
    with cells_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(cells)
    write_json(cells_json, cells)

    step6b_dem = json.loads(STEP6B_DEM_AVAILABILITY.read_text(encoding="utf-8"))
    required_coverage = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_geospatial_inventory_csv": GEOSPATIAL_INVENTORY_CSV.as_posix(),
        "source_dem_availability_json": STEP6B_DEM_AVAILABILITY.as_posix(),
        "sample_count": len(rows),
        "unique_event_locations": event_locations,
        "unique_event_location_count": len(event_locations),
        "crs": crs_values,
        "total_bounds": {
            "west": total_west,
            "south": total_south,
            "east": total_east,
            "north": total_north,
            "min_longitude": total_west,
            "max_longitude": total_east,
            "min_latitude": total_south,
            "max_latitude": total_north,
        },
        "required_1_degree_cell_count": len(cells),
        "required_dem_cells_csv": cells_csv.as_posix(),
        "required_dem_cells_json": cells_json.as_posix(),
        "cells_per_split": {split: len(values) for split, values in sorted(per_split.items())},
        "cells_per_event_location": {
            event: len(values) for event, values in sorted(per_event.items())
        },
        "step6b_dem_source_available": bool(step6b_dem.get("dem_source_available", False)),
    }
    write_json(RUN_DIR / "inventory" / "step6b2_required_dem_coverage.json", required_coverage)

    post_scan = rescan_dem_sources()
    write_json(RUN_DIR / "inventory" / "post_6b_dem_rescan.json", post_scan)
    if not post_scan["dem_source_available"]:
        write_verified_empty(post_scan)

    print(f"samples={len(rows)}")
    print(f"event_locations={len(event_locations)}")
    print(f"required_cells={len(cells)}")
    print(f"dem_source_available={post_scan['dem_source_available']}")
    print(f"required_cells_csv={cells_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
