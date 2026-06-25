"""Debug script: print one SegMAN training batch's shapes and statistics.

Usage:
    python experiments_cvpr/segman/debug_one_batch.py --config configs/segman/segman_dice_ce_topo.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

SEGMAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEGMAN_ROOT.parents[1]
for p in (str(SEGMAN_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from step6c_v3_train import TopographyDataModule  # noqa: E402
from train_segman import InputAssembler, get_target  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    with args.config.open("r", encoding="utf-8-sig") as fh:
        config = yaml.safe_load(fh)
    dem_map_file = config.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file:
        with open(dem_map_file, encoding="utf-8") as f:
            config.setdefault("dem", {})["dem_tile_id_map"] = json.load(f).get("mapping", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = TopographyDataModule(config, batch_size=int(config["trainer"]["batch_size"]))
    dm.setup("fit")
    loader = dm.train_dataloader()
    assembler = InputAssembler(config, device)

    batch = next(iter(loader))
    print("=" * 70)
    print(f"config        : {args.config.name}   loss_mode={config['loss']['mode']}")
    print("-" * 70)
    print("image modalities (raw, per modality):")
    for m, t in batch["image"].items():
        print(f"  {m:8s} shape={tuple(t.shape)} dtype={t.dtype} "
              f"min={t.float().min():.2f} max={t.float().max():.2f}")
    x = assembler({k: v.to(device) for k, v in batch["image"].items()})
    print(f"assembled input  : shape={tuple(x.shape)}  (15ch normalized)  "
          f"mean={x.mean():.3f} std={x.std():.3f}")

    target = get_target({"mask": batch["mask"]})
    uniq = torch.unique(target)
    ignored = int((target == int(config["loss"].get("ignore_index", -1))).sum())
    print(f"mask             : shape={tuple(target.shape)} dtype={target.dtype} "
          f"unique_values={uniq.tolist()}")
    print(f"ignored pixels   : {ignored} / {target.numel()} "
          f"({100.0 * ignored / target.numel():.2f}%)  ignore_index={config['loss'].get('ignore_index', -1)}")
    water = int((target == 1).sum())
    print(f"water pixels     : {water} ({100.0 * water / target.numel():.2f}%)")

    topo = batch.get("topography")
    if topo is not None:
        finite = torch.isfinite(topo)
        print(f"DEM/topography   : shape={tuple(topo.shape)} dtype={topo.dtype} "
              f"min={topo[finite].min():.2f} max={topo[finite].max():.2f} "
              f"nonfinite={int((~finite).sum())}")
    else:
        print("DEM/topography   : (not loaded for this loss mode)")

    # spatial-dimension consistency check
    H, W = target.shape[-2:]
    ok = all(t.shape[-2:] == (H, W) for t in batch["image"].values())
    ok = ok and (topo is None or topo.shape[-2:] == (H, W))
    print("-" * 70)
    print(f"spatial dims match (image == mask == DEM): {ok}  (H={H}, W={W})")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
