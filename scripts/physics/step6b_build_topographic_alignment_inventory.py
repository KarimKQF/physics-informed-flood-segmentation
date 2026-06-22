from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio


REPO_ROOT = Path("C:/Users/Karim/Desktop/flood-segmentation-training/physics-informed-flood-segmentation")
RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b_topographic_alignment_validation"
)
STEP5E_MANIFEST_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step5e_tiny_unetdecoder_baseline/manifests"
)
INDEX_CSV = STEP5E_MANIFEST_DIR / "sen1floods11_handlabeled_index_e_paths.csv"
STEP5N_TOPO_INVENTORY = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step5n_baseline_freeze_and_physics_loss_prep/data_availability/"
    "topographic_inputs_inventory.json"
)
DERIVED_TOPO_ROOT = Path("E:/flood_research/data/derived/sen1floods11_topography")
DATA_ROOT = Path("E:/flood_research/data")

EXCLUDED_TILE_IDS = {
    "Ghana_234935",
    "Ghana_26376",
    "Ghana_277",
    "Ghana_5079",
    "Ghana_83483",
}

SPLIT_FILES = {
    "train": STEP5E_MANIFEST_DIR / "flood_train_step5e_filtered.txt",
    "valid": STEP5E_MANIFEST_DIR / "flood_valid_step5e_filtered.txt",
    "test": STEP5E_MANIFEST_DIR / "flood_test_step5e_filtered.txt",
    "bolivia": STEP5E_MANIFEST_DIR / "flood_bolivia_step5e_filtered.txt",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_index() -> dict[str, dict[str, str]]:
    with INDEX_CSV.open("r", newline="", encoding="utf-8") as handle:
        return {row["tile_id"]: row for row in csv.DictReader(handle)}


def raster_metadata(path: str) -> dict[str, Any]:
    raster_path = Path(path)
    result: dict[str, Any] = {
        "path": path,
        "exists": raster_path.exists(),
        "readable": False,
        "crs": None,
        "transform": None,
        "height": None,
        "width": None,
        "count": None,
        "dtypes": None,
        "nodata": None,
        "bounds": None,
        "error": None,
    }
    if not raster_path.exists():
        result["error"] = "missing_file"
        return result
    try:
        with rasterio.open(raster_path) as ds:
            result.update(
                {
                    "readable": True,
                    "crs": str(ds.crs) if ds.crs else None,
                    "transform": [round(float(v), 12) for v in ds.transform],
                    "height": ds.height,
                    "width": ds.width,
                    "count": ds.count,
                    "dtypes": list(ds.dtypes),
                    "nodata": json_safe(ds.nodata),
                    "bounds": [
                        round(float(ds.bounds.left), 9),
                        round(float(ds.bounds.bottom), 9),
                        round(float(ds.bounds.right), 9),
                        round(float(ds.bounds.top), 9),
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def metadata_match(first: dict[str, Any], second: dict[str, Any]) -> bool:
    keys = ("crs", "transform", "height", "width")
    return all(first.get(key) == second.get(key) for key in keys)


def build_geospatial_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = read_index()
    rows: list[dict[str, Any]] = []
    split_sizes: dict[str, int] = {}

    for split, split_file in SPLIT_FILES.items():
        ids = [tile_id for tile_id in read_split_ids(split_file) if tile_id not in EXCLUDED_TILE_IDS]
        split_sizes[split] = len(ids)
        for tile_id in ids:
            source = index.get(tile_id)
            if source is None:
                rows.append(
                    {
                        "tile_id": tile_id,
                        "split": split,
                        "event_location": tile_id.split("_", 1)[0],
                        "status": "missing_index_row",
                    }
                )
                continue

            s1 = raster_metadata(source["sentinel1_path"])
            s2 = raster_metadata(source["sentinel2_path"])
            label = raster_metadata(source["hand_labeled_mask_path"])
            s1_matches_label = metadata_match(s1, label)
            s2_matches_label = metadata_match(s2, label)
            status = (
                "ok"
                if label["readable"] and s1["readable"] and s2["readable"] and s1_matches_label and s2_matches_label
                else "metadata_mismatch_or_unreadable"
            )
            rows.append(
                {
                    "tile_id": tile_id,
                    "split": split,
                    "event_location": source.get("event_location") or tile_id.split("_", 1)[0],
                    "s1_path": source["sentinel1_path"],
                    "s1_crs": s1["crs"],
                    "s1_transform": json.dumps(s1["transform"]),
                    "s1_height": s1["height"],
                    "s1_width": s1["width"],
                    "s1_count": s1["count"],
                    "s1_dtypes": ";".join(s1["dtypes"] or []),
                    "s1_nodata": s1["nodata"],
                    "s1_bounds": json.dumps(s1["bounds"]),
                    "s2_path": source["sentinel2_path"],
                    "s2_crs": s2["crs"],
                    "s2_transform": json.dumps(s2["transform"]),
                    "s2_height": s2["height"],
                    "s2_width": s2["width"],
                    "s2_count": s2["count"],
                    "s2_dtypes": ";".join(s2["dtypes"] or []),
                    "s2_nodata": s2["nodata"],
                    "s2_bounds": json.dumps(s2["bounds"]),
                    "label_path": source["hand_labeled_mask_path"],
                    "label_crs": label["crs"],
                    "label_transform": json.dumps(label["transform"]),
                    "label_height": label["height"],
                    "label_width": label["width"],
                    "label_count": label["count"],
                    "label_dtypes": ";".join(label["dtypes"] or []),
                    "label_nodata": label["nodata"],
                    "label_bounds": json.dumps(label["bounds"]),
                    "s1_matches_label_grid": s1_matches_label,
                    "s2_matches_label_grid": s2_matches_label,
                    "status": status,
                    "notes": "",
                }
            )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_csv": INDEX_CSV.as_posix(),
        "split_files": {split: path.as_posix() for split, path in SPLIT_FILES.items()},
        "excluded_tile_ids": sorted(EXCLUDED_TILE_IDS),
        "split_sizes": split_sizes,
        "total_rows": len(rows),
        "ok_rows": len(ok_rows),
        "problem_rows": len(rows) - len(ok_rows),
        "all_s1_s2_match_label_grid": len(ok_rows) == len(rows),
        "unique_crs": sorted({row.get("label_crs") for row in rows if row.get("label_crs")}),
        "unique_label_shapes": sorted(
            {
                f"{row.get('label_height')}x{row.get('label_width')}"
                for row in rows
                if row.get("label_height") and row.get("label_width")
            }
        ),
    }
    return rows, summary


def write_geospatial_inventory(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = RUN_DIR / "inventory" / "sen1floods11_geospatial_inventory.csv"
    json_path = RUN_DIR / "inventory" / "sen1floods11_geospatial_inventory.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, {"summary": summary, "rows": rows})


def build_inventory_review() -> dict[str, Any]:
    step5n = json.loads(STEP5N_TOPO_INVENTORY.read_text(encoding="utf-8"))
    topo = step5n.get("topographic_inputs", {})
    unavailable = [
        key
        for key in ["dem", "srtm", "hand_height_above_nearest_drainage", "slope", "flow_direction", "elevation"]
        if not topo.get(key, {}).get("available", False)
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_inventory": STEP5N_TOPO_INVENTORY.as_posix(),
        "step6a_report": (REPO_ROOT / "reports" / "STEP_6A_physics_topographic_loss_implementation_report.md").as_posix(),
        "main_stub": (
            REPO_ROOT / "configs" / "physics_loss" / "terramind_l_upernet_topographic_loss_stub.yaml"
        ).as_posix(),
        "control_stub": (
            REPO_ROOT / "configs" / "physics_loss" / "terramind_base_unetdecoder_topographic_loss_control_stub.yaml"
        ).as_posix(),
        "dem_or_hand_topography_available": False,
        "permanent_water_available_but_not_topography": bool(
            topo.get("permanent_water_mask", {}).get("available", False)
        ),
        "geospatial_metadata_available": bool(topo.get("geospatial_metadata", {}).get("available", False)),
        "available_topographic_paths": [],
        "already_aligned_to_sen1floods11": False,
        "unavailable_inputs": unavailable,
        "blockers": [
            topo.get(key, {}).get("blocker")
            for key in unavailable
            if topo.get(key, {}).get("blocker")
        ],
        "conclusion": (
            "No local DEM/SRTM/HAND/elevation source is available. Sen1Floods11 GeoTIFF metadata "
            "can support alignment once a DEM/HAND source is provided."
        ),
    }


def is_false_hand_label_path(path: Path) -> bool:
    lowered = path.as_posix().lower()
    return any(
        marker in lowered
        for marker in [
            "handlabeled",
            "labelhand",
            "s1hand",
            "s2hand",
            "jrcwaterhand",
            "s1otsulabelhand",
        ]
    )


def build_dem_source_availability() -> dict[str, Any]:
    raster_extensions = {".tif", ".tiff", ".vrt", ".img", ".bil", ".dem"}
    keywords = [
        "dem",
        "srtm",
        "copernicus",
        "elevation",
        "dtm",
        "dsm",
        "hand",
        "slope",
        "flow_direction",
        "flowdir",
        "topography",
        "topo",
    ]
    candidates: list[str] = []
    scanned_files = 0
    for path in DATA_ROOT.rglob("*"):
        if path.is_file():
            scanned_files += 1
            lowered = path.as_posix().lower()
            if path.suffix.lower() in raster_extensions and any(key in lowered for key in keywords):
                if "hand" in lowered and is_false_hand_label_path(path):
                    continue
                candidates.append(path.as_posix())

    derived_dirs = {
        "root": DERIVED_TOPO_ROOT.as_posix(),
        "exists": DERIVED_TOPO_ROOT.exists(),
        "children": [
            child.as_posix()
            for child in DERIVED_TOPO_ROOT.iterdir()
            if DERIVED_TOPO_ROOT.exists() and child.is_dir()
        ]
        if DERIVED_TOPO_ROOT.exists()
        else [],
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root_scanned": DATA_ROOT.as_posix(),
        "scanned_file_count": scanned_files,
        "candidate_keywords": keywords,
        "candidate_raster_count": len(candidates),
        "candidate_rasters": candidates[:100],
        "dem_source_available": len(candidates) > 0,
        "derived_topography_location": derived_dirs,
        "permanent_water_note": (
            "JRC permanent-water rasters exist in Sen1Floods11 but are not DEM/HAND/topography "
            "and cannot drive TopographicInconsistencyLoss."
        ),
        "blocker": None
        if candidates
        else "No DEM/SRTM/HAND/elevation raster source found locally under E:/flood_research/data.",
    }


def write_manifest_schema() -> None:
    schema = {
        "description": "Manifest schema for aligned Sen1Floods11 topographic inputs.",
        "fields": [
            {"name": "tile_id", "type": "string", "required": True},
            {"name": "split", "type": "string", "required": True},
            {"name": "event_location", "type": "string", "required": True},
            {"name": "s1_path", "type": "path", "required": True},
            {"name": "s2_path", "type": "path", "required": True},
            {"name": "label_path", "type": "path", "required": True},
            {"name": "topography_path", "type": "path", "required": True},
            {"name": "topography_type", "type": "string", "required": True, "examples": ["dem", "hand"]},
            {"name": "crs", "type": "string", "required": True},
            {"name": "height", "type": "integer", "required": True},
            {"name": "width", "type": "integer", "required": True},
            {"name": "finite_ratio", "type": "float", "required": True},
            {"name": "nodata_ratio", "type": "float", "required": True},
            {"name": "status", "type": "string", "required": True},
            {"name": "notes", "type": "string", "required": False},
        ],
        "guards": {
            "raw_data_modified": False,
            "requires_topographic_alignment_validated_before_training": True,
        },
    }
    write_json(RUN_DIR / "manifests" / "topography_manifest_schema.json", schema)


def main() -> int:
    for subdir in ["reports", "logs", "metrics", "figures", "inventory", "scripts", "sample_outputs", "manifests"]:
        (RUN_DIR / subdir).mkdir(parents=True, exist_ok=True)

    rows, summary = build_geospatial_inventory()
    write_geospatial_inventory(rows, summary)
    write_json(RUN_DIR / "inventory" / "existing_topographic_inventory_review.json", build_inventory_review())
    write_json(RUN_DIR / "inventory" / "dem_source_availability.json", build_dem_source_availability())
    write_manifest_schema()
    print(f"geospatial_rows={summary['total_rows']}")
    print(f"geospatial_ok_rows={summary['ok_rows']}")
    print(f"inventory_dir={RUN_DIR / 'inventory'}")
    print(f"manifest_schema={RUN_DIR / 'manifests' / 'topography_manifest_schema.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
