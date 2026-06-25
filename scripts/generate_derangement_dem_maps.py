"""
Regenerate DEM shuffle mappings as strict derangements (0 self-maps).

Uses Sattolo's algorithm: deterministic single-pass derangement.
  - For each i from n-1 down to 1: swap items[i] with items[j], j = rng.randrange(0, i)
  - Guarantees no fixed points (derangement) AND produces a single cycle.
  - Deterministic from seed; always succeeds in one pass.

DEM shuffle seed formula (unchanged): manifest_seed + 10000.
Overwrites dem_shuffle_map_n50_seed{S}.json in place.
Does NOT touch manifests or any other config file.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "manifests" / "terramind_baseline" / "low_data_multiseed"
SEEDS = [0, 1, 2, 3, 42]
DEM_SHUFFLE_SEED_OFFSET = 10000
DEM_ROOT = Path("E:/flood_research/data/derived/sen1floods11_topography/dem_aligned")
SPLIT_NAMES = ("train", "valid", "test", "Bolivia")


def sattolo_derangement(items: list, seed: int) -> list:
    """
    Sattolo's algorithm: permutation guaranteed to be a single cycle (hence a derangement).
    Deterministic given (items, seed). Always produces exactly 0 fixed points.

    Algorithm:
        for i from n-1 down to 1:
            j = randrange(0, i)   # strictly less than i
            swap items[i] and items[j]
    """
    result = list(items)
    rng = random.Random(seed)
    n = len(result)
    for i in range(n - 1, 0, -1):
        j = rng.randrange(0, i)  # 0 <= j < i  (strictly less than i)
        result[i], result[j] = result[j], result[i]
    return result


def verify_derangement(original: list, permuted: list) -> dict:
    assert len(original) == len(permuted), "Length mismatch"
    self_maps = [orig for orig, perm in zip(original, permuted) if orig == perm]
    targets = list(permuted)
    is_bijective = sorted(targets) == sorted(original)
    is_single_cycle = _is_single_cycle(original, permuted)
    return {
        "n": len(original),
        "self_maps": len(self_maps),
        "self_map_items": self_maps,
        "is_bijective": is_bijective,
        "is_single_cycle": is_single_cycle,
        "valid_derangement": len(self_maps) == 0 and is_bijective,
    }


def _is_single_cycle(original: list, permuted: list) -> bool:
    """Check that the permutation is a single cycle (Sattolo property)."""
    index = {v: i for i, v in enumerate(original)}
    visited = set()
    current = original[0]
    while current not in visited:
        visited.add(current)
        pos = index[current]
        current = permuted[pos]
    return len(visited) == len(original)


def check_dem_files_exist(tile_ids: list, split_name: str = "train") -> dict:
    """Check that aligned DEM files exist for the given tile_ids."""
    missing = []
    for tile_id in tile_ids:
        path = DEM_ROOT / f"{split_name}_{tile_id}_copernicus_glo30_dem_aligned.tif"
        if not path.exists():
            missing.append(str(path))
    return {"n_checked": len(tile_ids), "n_missing": len(missing), "missing": missing}


def main() -> None:
    validation_records = {}

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")

        # Load manifest
        manifest_path = MANIFEST_DIR / f"flood_train_low_data_n50_seed{seed}.txt"
        assert manifest_path.exists(), f"Manifest not found: {manifest_path}"
        tile_ids = sorted(
            line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        assert len(tile_ids) == 50, f"Expected 50 tiles, got {len(tile_ids)}"
        print(f"  Manifest: {len(tile_ids)} tiles (unchanged)")

        # Generate derangement
        dem_seed = seed + DEM_SHUFFLE_SEED_OFFSET
        permuted = sattolo_derangement(tile_ids, dem_seed)

        # Verify
        verification = verify_derangement(tile_ids, permuted)
        assert verification["valid_derangement"], f"Not a valid derangement: {verification}"
        print(f"  Self-maps:    {verification['self_maps']}/50  (was non-zero for seeds 0,1,2,3)")
        print(f"  Bijective:    {verification['is_bijective']}")
        print(f"  Single cycle: {verification['is_single_cycle']}")

        # Check DEM files exist (targets are still within the same train split)
        dem_check = check_dem_files_exist(permuted, split_name="train")
        if dem_check["n_missing"] > 0:
            print(f"  WARNING: {dem_check['n_missing']} DEM files missing for targets")
        else:
            print(f"  DEM files:    all {dem_check['n_checked']} targets exist")

        # Build mapping dict: real_tile_id -> dem_tile_id_to_load
        mapping = {real: permuted[i] for i, real in enumerate(tile_ids)}

        # Write JSON (overwrite existing file)
        out_path = MANIFEST_DIR / f"dem_shuffle_map_n50_seed{seed}.json"
        payload = {
            "purpose": "Shuffled DEM control for E2 experiment — strict derangement",
            "algorithm": "Sattolo's algorithm (single-cycle permutation, guaranteed derangement)",
            "manifest_seed": seed,
            "dem_shuffle_seed": dem_seed,
            "dem_shuffle_seed_formula": "manifest_seed + 10000",
            "n_tiles": 50,
            "n_self_maps": 0,
            "is_bijective": verification["is_bijective"],
            "is_single_cycle": verification["is_single_cycle"],
            "note": (
                "Each training sample's tile_id maps to a DIFFERENT sample's DEM. "
                "Single-cycle property means every DEM tile is used exactly once "
                "and the mapping chains through all 50 samples before returning to start. "
                "Topo metrics at eval time should use real DEM for physical coherence measurement."
            ),
            "mapping": mapping,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        h = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
        print(f"  Written: {out_path} (sha256[:16]={h})")

        validation_records[str(seed)] = {
            "manifest_path": str(manifest_path),
            "dem_map_path": str(out_path),
            "dem_shuffle_seed": dem_seed,
            "n_tiles": 50,
            "self_maps": 0,
            "is_bijective": verification["is_bijective"],
            "is_single_cycle": verification["is_single_cycle"],
            "valid_derangement": True,
            "dem_files_checked": dem_check["n_checked"],
            "dem_files_missing": dem_check["n_missing"],
            "sha256_16": h,
        }

    # Write validation summary
    val_path = MANIFEST_DIR / "dem_shuffle_derangement_validation.json"
    val_payload = {
        "description": "Validation of DEM shuffle derangement mappings for E1+E2 experiment",
        "algorithm": "Sattolo's algorithm — single-cycle derangement, deterministic from seed",
        "derangement_property": "No sample receives its own DEM (0 self-maps, verified per seed)",
        "bijection_property": "Each DEM target used exactly once (permutation, not just injection)",
        "seeds": SEEDS,
        "all_valid": all(r["valid_derangement"] for r in validation_records.values()),
        "records": validation_records,
    }
    val_path.write_text(json.dumps(val_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nValidation JSON: {val_path}")

    # Verify configs still point to correct paths
    print("\n--- Config path verification ---")
    config_dir = REPO_ROOT / "configs" / "multiseed_n50"
    for seed in SEEDS:
        cfg = config_dir / f"n50_seed{seed}_physics_shuffled_dem_lambda05.yaml"
        expected_map = str(MANIFEST_DIR / f"dem_shuffle_map_n50_seed{seed}.json").replace("\\", "/")
        text = cfg.read_text(encoding="utf-8")
        if expected_map.replace("/", "\\") in text or expected_map in text:
            print(f"  seed={seed}: config OK -> {expected_map.split('/')[-1]}")
        else:
            print(f"  seed={seed}: WARNING config may not point to correct map file")
            print(f"    Expected: {expected_map}")

    print("\nDone. Manifests unchanged. No runs launched.")


if __name__ == "__main__":
    main()
