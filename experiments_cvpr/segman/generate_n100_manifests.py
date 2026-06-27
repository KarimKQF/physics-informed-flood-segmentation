"""
Generate N=100 training manifests and DEM shuffle maps for seeds 0, 1, 2.

Protocol mirrors the existing N=50 manifests exactly:
  - Source: manifests/terramind_baseline/flood_train_data.txt  (251 tiles)
  - Algorithm: random.Random(seed).shuffle(all_tiles); take first 100; write sorted
  - Sanity check: first 50 of N=100 must match existing N=50 manifest for each seed
  - DEM shuffle: Sattolo's algorithm on N=100 tiles, seed = manifest_seed + 10000
  - Output: manifests/terramind_baseline/low_data_multiseed_n100/
    flood_train_low_data_n100_seed{s}.txt
    dem_shuffle_map_n100_seed{s}.json
    selection_summary_n100.json

Usage:
    python experiments_cvpr/segman/generate_n100_manifests.py
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_MANIFEST = REPO_ROOT / "manifests" / "terramind_baseline" / "flood_train_data.txt"
N50_MANIFEST_DIR = REPO_ROOT / "manifests" / "terramind_baseline" / "low_data_multiseed"
OUT_DIR = REPO_ROOT / "manifests" / "terramind_baseline" / "low_data_multiseed_n100"

SEEDS = [0, 1, 2]
N = 100

VAL_SPLIT   = "E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_valid_step5e_filtered.txt"
TEST_SPLIT  = "E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt"
BOL_SPLIT   = "E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_bolivia_step5e_filtered.txt"


def sattolo_derangement(items: list[str], seed: int) -> dict[str, str]:
    """Sattolo's algorithm — guaranteed single-cycle derangement (no fixed points)."""
    lst = list(items)
    rng = random.Random(seed)
    n = len(lst)
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i - 1)      # j in [0, i-1] → guarantees no fixed point
        lst[i], lst[j] = lst[j], lst[i]
    mapping = {items[i]: lst[i] for i in range(n)}
    # Validate
    for k, v in mapping.items():
        assert k != v, f"Self-map detected: {k} -> {v}"
    return mapping


def sha256_16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main() -> None:
    # ── Load source manifest ──────────────────────────────────────────────
    all_tiles = [line.strip() for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    print(f"Source manifest: {SOURCE_MANIFEST}")
    print(f"Total tiles: {len(all_tiles)}")
    assert len(all_tiles) == 251, f"Expected 251 tiles, got {len(all_tiles)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "protocol": "SegMAN N=100 diagnostic — 3 seeds x 3 conditions",
        "seeds": SEEDS,
        "N": N,
        "source_train_manifest": str(SOURCE_MANIFEST),
        "source_train_count": len(all_tiles),
        "sampling_algorithm": f"random.Random(seed).shuffle(full_train_ids); take first {N}; write sorted",
        "dem_shuffle_seed_formula": "manifest_seed + 10000",
        "valid_split_unchanged": VAL_SPLIT,
        "test_split_unchanged": TEST_SPLIT,
        "bolivia_split_unchanged": BOL_SPLIT,
        "note": f"N=100 is a superset of N=50 (same seed): positions 0-49 in shuffled order are identical to the N=50 manifest.",
        "subsets": {},
        "dem_shuffle_maps": {},
    }

    for seed in SEEDS:
        print(f"\n-- Seed {seed} --")

        # ── Sample N=100 ─────────────────────────────────────────────────
        tiles_copy = list(all_tiles)
        rng = random.Random(seed)
        rng.shuffle(tiles_copy)
        selected_shuffled = tiles_copy[:N]
        selected_sorted   = sorted(selected_shuffled)

        # ── Sanity check: first 50 must match existing N=50 manifest ─────
        n50_path = N50_MANIFEST_DIR / f"flood_train_low_data_n50_seed{seed}.txt"
        n50_tiles = sorted([
            line.strip() for line in n50_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ])
        n50_expected = sorted(tiles_copy[:50])
        assert n50_expected == n50_tiles, (
            f"Seed {seed}: first 50 of N=100 shuffle do NOT match existing N=50 manifest!\n"
            f"  Expected: {n50_expected[:5]}...\n"
            f"  Got:      {n50_tiles[:5]}..."
        )
        print(f"  OK: First 50 match existing N=50 seed{seed} manifest")
        n100_extra = sorted(set(selected_sorted) - set(n50_tiles))
        print(f"  +50 new tiles: {n100_extra[:5]}... (total new: {len(n100_extra)})")

        # ── Write manifest ────────────────────────────────────────────────
        manifest_path = OUT_DIR / f"flood_train_low_data_n100_seed{seed}.txt"
        manifest_path.write_text("\n".join(selected_sorted) + "\n", encoding="utf-8")
        print(f"  Wrote: {manifest_path} ({len(selected_sorted)} tiles)")

        content_hash = sha256_16("\n".join(selected_sorted) + "\n")

        summary["subsets"][str(seed)] = {
            "path": str(manifest_path),
            "count": len(selected_sorted),
            "sha256_16": content_hash,
            "shuffled_order_first_100": selected_shuffled,
            "tile_ids_sorted": selected_sorted,
            "n50_overlap": n50_tiles,
            "n100_extra_tiles": n100_extra,
        }

        # ── DEM shuffle map (Sattolo) ─────────────────────────────────────
        dem_seed = seed + 10000
        mapping = sattolo_derangement(selected_shuffled, dem_seed)
        n_self = sum(1 for k, v in mapping.items() if k == v)
        assert n_self == 0, f"Sattolo produced {n_self} self-maps for seed {seed}"

        shuffle_map = {
            "purpose": f"Shuffled DEM control for N=100 diagnostic — strict Sattolo derangement, seed{seed}",
            "algorithm": "Sattolo's algorithm (single-cycle permutation, guaranteed derangement)",
            "manifest_seed": seed,
            "dem_shuffle_seed": dem_seed,
            "dem_shuffle_seed_formula": "manifest_seed + 10000",
            "n_tiles": N,
            "n_self_maps": 0,
            "is_bijective": True,
            "note": (
                f"Each training tile maps to a DIFFERENT tile's DEM. "
                "Eval-time topo metrics use the real DEM."
            ),
            "mapping": mapping,
        }

        shuffle_path = OUT_DIR / f"dem_shuffle_map_n100_seed{seed}.json"
        shuffle_path.write_text(json.dumps(shuffle_map, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
        print(f"  Wrote DEM shuffle map: {shuffle_path} (0 self-maps verified)")

        summary["dem_shuffle_maps"][str(seed)] = {
            "path": str(shuffle_path),
            "dem_shuffle_seed": dem_seed,
            "n_self_maps": 0,
        }

    # ── Write summary ─────────────────────────────────────────────────────
    summary_path = OUT_DIR / "selection_summary_n100.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")
    print("\nAll N=100 manifests and DEM shuffle maps generated successfully.")


if __name__ == "__main__":
    main()
