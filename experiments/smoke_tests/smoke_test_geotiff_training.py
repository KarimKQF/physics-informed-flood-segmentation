from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.data import GeoTIFFDataset, geotiff_collate_fn
from urban_runoff.losses import BinaryTopographicGradientLoss, MaskedBCEWithLogitsLoss
from urban_runoff.models import SimpleSegmentationCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test training on a GeoTIFF manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--use-dem", action="store_true")
    parser.add_argument("--max-samples", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(2026)
    if not args.manifest.exists():
        print(f"[ERROR] Manifest does not exist: {args.manifest}")
        return 1

    try:
        dataset = GeoTIFFDataset(
            args.manifest,
            max_samples=args.max_samples,
            require_dem=args.use_dem,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=min(2, len(dataset)),
            shuffle=False,
            collate_fn=geotiff_collate_fn,
        )
        batch = next(iter(dataloader))
    except Exception as exc:
        print(f"[ERROR] Could not load GeoTIFFDataset: {exc}")
        return 1

    image = batch["image"]
    mask = batch["mask"]
    valid_mask = batch["valid_mask"]
    dem = batch["dem"]
    if (
        not isinstance(image, Tensor)
        or not isinstance(mask, Tensor)
        or not isinstance(valid_mask, Tensor)
    ):
        print("[ERROR] image, mask and valid_mask must be tensors")
        return 1

    print(f"[OK] GeoTIFFDataset loaded: {len(dataset)} samples")
    print(f"[INFO] image shape: {tuple(image.shape)}")
    print(f"[INFO] mask shape: {tuple(mask.shape)}")
    print("[OK] valid_mask loaded")
    print(f"[INFO] valid_mask shape: {tuple(valid_mask.shape)}")
    print(f"[INFO] mask unique values after binarization: {torch.unique(mask).tolist()}")
    print(f"[INFO] valid_mask unique values: {torch.unique(valid_mask).tolist()}")
    print(f"[INFO] valid pixels ratio: {float(valid_mask.float().mean()):.6f}")
    if dem is None:
        print("[INFO] DEM: not available")
    else:
        print("[OK] DEM loaded")
        print(f"[INFO] dem shape: {tuple(dem.shape)}")
        print(f"[INFO] dem min/max: {float(dem.min()):.6f} / {float(dem.max()):.6f}")

    model = SimpleSegmentationCNN(in_channels=image.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    print(f"[INFO] logits shape: {tuple(logits.shape)}")

    masked_bce = MaskedBCEWithLogitsLoss()
    if args.use_dem:
        if dem is None or not isinstance(dem, Tensor):
            print("[ERROR] --use-dem was set but no DEM tensor is available in the batch")
            return 1
        topographic = BinaryTopographicGradientLoss()
        classical_loss = masked_bce(logits, mask, valid_mask)
        topographic_loss = topographic(logits=logits, target=mask, dem=dem, valid_mask=valid_mask)
        loss = classical_loss + 0.1 * topographic_loss
        print(f"[INFO] Masked BCE loss: {float(classical_loss.detach()):.6f}")
        print(f"[INFO] Topographic loss: {float(topographic_loss.detach()):.6f}")
        print(f"[INFO] Total loss: {float(loss.detach()):.6f}")
    else:
        loss = masked_bce(logits, mask, valid_mask)
        print(f"[INFO] Masked BCE loss: {float(loss.detach()):.6f}")

    if not torch.isfinite(loss):
        print("[ERROR] Loss is not finite")
        return 1

    loss.backward()
    optimizer.step()
    print("[OK] Backward pass successful")
    print("[OK] Optimizer step successful")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
