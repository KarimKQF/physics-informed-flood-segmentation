"""
Generate N=50 training manifests for multi-seed E1+E2 experiment.

Algorithm: Python random.Random(seed).shuffle(full_train_ids); take first 50.
Same algorithm as seed42 (selection_summary_seed42.json).

Seeds: 0, 1, 2, 3, 42 (seed42 already exists but is regenerated for verification).
Output: manifests/terramind_baseline/low_data_multiseed/flood_train_low_data_n50_seed{S}.txt
        manifests/terramind_baseline/low_data_multiseed/selection_summary_multiseed.json

Also generates shuffled DEM mappings (DEM assigned to wrong sample) per seed.
Shuffled DEM seed = manifest_seed + 10000 (avoids overlap).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_TRAIN = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step5e_tiny_unetdecoder_baseline/manifests/flood_train_step5e_filtered.txt"
)
LABEL_ROOT = Path(
    "E:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/"
    "HandLabeled/LabelHand"
)
OUT_DIR = REPO_ROOT / "manifests" / "terramind_baseline" / "low_data_multiseed"
SEEDS = [0, 1, 2, 3, 42]
N = 50
DEM_SHUFFLE_SEED_OFFSET = 10000  # manifest_seed + offset = DEM shuffle seed


def manifest_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def water_pixel_stats(tile_ids: list[str], label_root: Path) -> dict:
    """Scan LabelHand TIFs to count water pixels per tile."""
    import numpy as np
    import rasterio  # only imported when called; safe to fail if rasterio absent

    total_water = 0
    total_valid = 0
    water_pcts: list[float] = []
    failures: list[str] = []

    for tile_id in tile_ids:
        # Try all known splits for the tile
        found = False
        for split in ("Bolivia", "Ghana", "India", "Mekong", "Nigeria",
                      "Pakistan", "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA"):
            if tile_id.startswith(split + "_"):
                pattern = f"*{tile_id}*_LabelHand.tif"
                matches = list(label_root.glob(pattern))
                if matches:
                    try:
                        with rasterio.open(matches[0]) as ds:
                            arr = ds.read(1)
                        valid = (arr != -1)
                        water = (arr == 1)
                        n_valid = int(valid.sum())
                        n_water = int(water.sum())
                        total_water += n_water
                        total_valid += n_valid
                        pct = 100.0 * n_water / n_valid if n_valid > 0 else 0.0
                        water_pcts.append(pct)
                        found = True
                    except Exception as e:
                        failures.append(f"{tile_id}: {e}")
                    break
        if not found:
            failures.append(f"{tile_id}: file not found in {label_root}")

    if not water_pcts:
        return {"error": "no labels loaded", "failures": failures}

    import statistics
    return {
        "total_water_pixels": total_water,
        "total_valid_pixels": total_valid,
        "mean_water_pct": round(statistics.mean(water_pcts), 4),
        "median_water_pct": round(statistics.median(water_pcts), 4),
        "min_water_pct": round(min(water_pcts), 4),
        "max_water_pct": round(max(water_pcts), 4),
        "tiles_with_stats": len(water_pcts),
        "failures": failures,
    }


def classify_no_water(tile_ids: list[str], label_root: Path) -> tuple[list[str], list[str]]:
    """Split tile_ids into water_positive and no_water based on label scan."""
    try:
        import rasterio
        import numpy as np
    except ImportError:
        # Fall back: count by event geography (same heuristic as seed42 summary)
        # Ghana = mostly no-water in training set
        no_water = [t for t in tile_ids if t.startswith("Ghana_")][:2]
        water = [t for t in tile_ids if t not in no_water]
        return water, no_water

    water_pos, no_water = [], []
    for tile_id in tile_ids:
        pattern = f"*{tile_id}*_LabelHand.tif"
        matches = list(label_root.glob(pattern))
        if not matches:
            water_pos.append(tile_id)
            continue
        try:
            with rasterio.open(matches[0]) as ds:
                arr = ds.read(1)
            has_water = bool((arr == 1).any())
            if has_water:
                water_pos.append(tile_id)
            else:
                no_water.append(tile_id)
        except Exception:
            water_pos.append(tile_id)
    return water_pos, no_water


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read full training manifest
    full_ids = [line.strip() for line in SOURCE_TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(full_ids) == 251, f"Expected 251 train IDs, got {len(full_ids)}"
    print(f"Loaded {len(full_ids)} full train IDs from {SOURCE_TRAIN}")

    summary: dict = {
        "protocol": "E1+E2 multi-seed N=50 manifests",
        "seeds": SEEDS,
        "N": N,
        "source_train_manifest": str(SOURCE_TRAIN),
        "source_train_count": len(full_ids),
        "sampling_algorithm": (
            f"random.Random(seed).shuffle(full_train_ids); take first {N}; "
            "manifests written sorted for diff stability."
        ),
        "dem_shuffle_seed_formula": "manifest_seed + 10000",
        "valid_split_unchanged": "E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_valid_step5e_filtered.txt",
        "test_split_unchanged": "E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt",
        "bolivia_split_unchanged": "E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_bolivia_step5e_filtered.txt",
        "subsets": {},
        "dem_shuffle_maps": {},
    }

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")

        # 1. Generate manifest
        rng = random.Random(seed)
        shuffled = list(full_ids)
        rng.shuffle(shuffled)
        selected = shuffled[:N]
        selected_sorted = sorted(selected)  # sorted for diff stability

        manifest_path = OUT_DIR / f"flood_train_low_data_n50_seed{seed}.txt"
        manifest_path.write_text("\n".join(selected_sorted) + "\n", encoding="utf-8")
        h = manifest_hash(manifest_path)
        print(f"  Manifest: {manifest_path} (sha256[:16]={h})")

        # Event composition
        event_counts: dict[str, int] = {}
        for tid in selected:
            event = tid.split("_")[0]
            event_counts[event] = event_counts.get(event, 0) + 1

        # Water/no-water classification
        try:
            water_pos, no_water = classify_no_water(selected, LABEL_ROOT)
        except Exception as e:
            print(f"  WARNING: label scan failed ({e}); falling back to no-water heuristic")
            no_water = [t for t in selected if t.startswith("Ghana_")]
            water_pos = [t for t in selected if t not in no_water]

        print(f"  water_positive={len(water_pos)}  no_water={len(no_water)}")

        # Water pixel stats (best-effort; skipped if rasterio unavailable)
        try:
            stats = water_pixel_stats(selected, LABEL_ROOT)
        except Exception as e:
            stats = {"error": str(e)}

        summary["subsets"][str(seed)] = {
            "path": str(manifest_path),
            "count": N,
            "sha256_16": h,
            "shuffled_order_first_50": shuffled[:N],
            "tile_ids_sorted": selected_sorted,
            "event_location_counts": event_counts,
            "water_tile_count": len(water_pos),
            "no_water_count": len(no_water),
            "water_stats": stats,
        }

        # 2. Generate shuffled DEM mapping
        dem_seed = seed + DEM_SHUFFLE_SEED_OFFSET
        dem_rng = random.Random(dem_seed)
        dem_shuffled = list(selected)  # same tiles, shuffled assignment
        dem_rng.shuffle(dem_shuffled)

        dem_map = {real_id: dem_shuffled[i] for i, real_id in enumerate(selected)}
        # Verify: no tile maps to itself would be ideal, but small N means some self-maps are unavoidable
        self_maps = sum(1 for k, v in dem_map.items() if k == v)
        print(f"  DEM shuffle seed={dem_seed}: {self_maps}/{N} self-maps (acceptable if small)")

        dem_map_path = OUT_DIR / f"dem_shuffle_map_n50_seed{seed}.json"
        dem_map_data = {
            "purpose": "Shuffled DEM control for E2 experiment",
            "manifest_seed": seed,
            "dem_shuffle_seed": dem_seed,
            "dem_shuffle_seed_formula": "manifest_seed + 10000",
            "n_tiles": N,
            "n_self_maps": self_maps,
            "note": (
                "training sample tile_id -> DEM tile_id to load. "
                "Maps correct sample to wrong DEM, breaking sample-topography correspondence "
                "while preserving DEM value distribution and spatial structure."
            ),
            "mapping": dem_map,
        }
        dem_map_path.write_text(json.dumps(dem_map_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  DEM map:  {dem_map_path}")

        summary["dem_shuffle_maps"][str(seed)] = {
            "path": str(dem_map_path),
            "dem_shuffle_seed": dem_seed,
            "n_self_maps": self_maps,
        }

    # Write summary
    summary_path = OUT_DIR / "selection_summary_multiseed.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary_path}")

    # Verify seed42 matches existing manifest
    existing_seed42 = REPO_ROOT / "manifests" / "terramind_baseline" / "low_data_seed42" / "flood_train_low_data_n50_seed42.txt"
    new_seed42 = OUT_DIR / "flood_train_low_data_n50_seed42.txt"
    if existing_seed42.exists() and new_seed42.exists():
        existing_ids = set(l.strip() for l in existing_seed42.read_text().splitlines() if l.strip())
        new_ids = set(l.strip() for l in new_seed42.read_text().splitlines() if l.strip())
        if existing_ids == new_ids:
            print("\n✓ seed42 manifest matches existing manifest exactly")
        else:
            diff = existing_ids.symmetric_difference(new_ids)
            print(f"\n⚠ seed42 manifest MISMATCH — symmetric diff: {diff}")

    print("\nDone. No training launched.")


if __name__ == "__main__":
    main()
