from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

MANIFEST_COLUMNS = ["sample_id", "image_path", "mask_path", "dem_path", "split"]


def lat_label(lat: int) -> str:
    return f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"


def lon_label(lon: int) -> str:
    return f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"


def srtm_tile_names(bounds: dict[str, float]) -> list[str]:
    left = bounds["left"]
    bottom = bounds["bottom"]
    right = bounds["right"]
    top = bounds["top"]
    lon_values = range(math.floor(left), math.ceil(right))
    lat_values = range(math.floor(bottom), math.ceil(top))
    names = []
    for lat in lat_values:
        for lon in lon_values:
            names.append(f"{lat_label(lat)}{lon_label(lon)}")
    return names


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not set(MANIFEST_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns: {MANIFEST_COLUMNS}")
        return list(reader)


def read_bounds(bounds_json: Path) -> dict[str, dict[str, Any]]:
    if not bounds_json.exists():
        raise FileNotFoundError(f"Bounds JSON does not exist: {bounds_json}")
    payload = json.loads(bounds_json.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    return {sample["sample_id"]: sample for sample in samples}


def sample_center(sample_bounds: dict[str, float]) -> tuple[float, float]:
    return (
        (sample_bounds["left"] + sample_bounds["right"]) / 2.0,
        (sample_bounds["bottom"] + sample_bounds["top"]) / 2.0,
    )


def squared_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def select_compact_samples(
    rows: list[dict[str, str]],
    bounds_by_sample: dict[str, dict[str, Any]],
    *,
    max_samples: int,
    max_unique_srtm_tiles: int,
) -> tuple[list[dict[str, str]], list[str]]:
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if max_unique_srtm_tiles <= 0:
        raise ValueError("--max-unique-srtm-tiles must be positive.")

    candidates: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["sample_id"]
        sample = bounds_by_sample.get(sample_id)
        if sample is None:
            raise ValueError(f"Bounds JSON has no sample entry for {sample_id}")
        tiles = set(srtm_tile_names(sample["bounds"]))
        if not tiles:
            raise ValueError(f"No SRTM tile found for {sample_id}")
        candidates.append(
            {
                "row": row,
                "sample_id": sample_id,
                "tiles": tiles,
                "center": sample_center(sample["bounds"]),
            }
        )

    best_selection: list[dict[str, Any]] = []
    best_tiles: set[str] = set()

    for seed in candidates:
        selected = [seed]
        selected_ids = {seed["sample_id"]}
        tile_union = set(seed["tiles"])

        while len(selected) < min(max_samples, len(candidates)):
            centroid = (
                sum(item["center"][0] for item in selected) / len(selected),
                sum(item["center"][1] for item in selected) / len(selected),
            )
            choices = []
            for candidate in candidates:
                if candidate["sample_id"] in selected_ids:
                    continue
                added_tiles = candidate["tiles"] - tile_union
                new_tile_count = len(tile_union | candidate["tiles"])
                if new_tile_count > max_unique_srtm_tiles:
                    continue
                choices.append(
                    (
                        len(added_tiles),
                        new_tile_count,
                        squared_distance(candidate["center"], centroid),
                        candidate["sample_id"],
                        candidate,
                    )
                )
            if not choices:
                break
            _, _, _, _, chosen = min(choices)
            selected.append(chosen)
            selected_ids.add(chosen["sample_id"])
            tile_union |= chosen["tiles"]

        if len(selected) > len(best_selection) or (
            len(selected) == len(best_selection) and len(tile_union) < len(best_tiles or tile_union)
        ):
            best_selection = selected
            best_tiles = tile_union

    if not best_selection:
        sorted_candidates = sorted(
            candidates,
            key=lambda item: (len(item["tiles"]), item["sample_id"]),
        )
        best_selection = sorted_candidates[:max_samples]
        best_tiles = set().union(*(item["tiles"] for item in best_selection))

    return [dict(item["row"], dem_path="") for item in best_selection], sorted(best_tiles)


def write_manifest(output_manifest: Path, rows: list[dict[str, str]]) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a geographically compact Sen1Floods11 subset manifest for DEM tests."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bounds-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-unique-srtm-tiles", type=int, default=16)
    args = parser.parse_args()

    try:
        rows = read_manifest(args.manifest)
        bounds_by_sample = read_bounds(args.bounds_json)
        selected_rows, unique_tiles = select_compact_samples(
            rows,
            bounds_by_sample,
            max_samples=args.max_samples,
            max_unique_srtm_tiles=args.max_unique_srtm_tiles,
        )
        write_manifest(args.output_manifest, selected_rows)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    selected_ids = [row["sample_id"] for row in selected_rows]
    print(f"[INFO] Selected samples: {selected_ids}")
    print(f"[INFO] Unique SRTM tiles required: {unique_tiles}")
    print(f"[OK] Compact manifest written to: {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
