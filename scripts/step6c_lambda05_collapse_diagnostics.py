"""
STEP 6C lambda=0.5 collapse diagnostics (READ-ONLY w.r.t. the failed run).

This script does NOT train, does NOT modify raw data, does NOT use DEM as model
input, does NOT overwrite the failed-run checkpoints/logs/configs. It performs:

  7.3 Prediction-distribution diagnostics (fresh pretrained init + collapsed ckpt).
  7.4 Loss parity vs STEP 5S-A (terratorch smp DiceLoss) on one real batch.
  7.6 Optimizer / gradient diagnostics (one controlled batch).
  7.7 Topographic-loss degeneracy probe (synthetic p_water=0.5 logits).

Output JSON:
  E:/.../runs/step6c_terramind_l_upernet_dice_topographic_lambda05/metrics/collapse_diagnostics.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402

import step6c_lambda05_train as t6c  # noqa: E402  (reuse the EXACT data/model path)
from losses.combined_loss import CombinedDicePhysicsLoss, _soft_dice_loss  # noqa: E402
from losses.physics_topographic_loss import TopographicInconsistencyLoss  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "step6c_terramind_l_upernet_dice_topographic_lambda05.yaml"
RUN_DIR = Path("E:/flood_research/experiments/terramind_baseline/runs/step6c_terramind_l_upernet_dice_topographic_lambda05")
LAST_CKPT = RUN_DIR / "checkpoints" / "last_checkpoint.pt"
BEST_CKPT = RUN_DIR / "checkpoints" / "best_checkpoint.pt"
OUT_JSON = RUN_DIR / "metrics" / "collapse_diagnostics.json"


def jsafe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): jsafe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsafe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if torch.is_tensor(v):
        v = v.detach().cpu()
        return jsafe(v.item() if v.numel() == 1 else v.tolist())
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def prediction_distribution(logits: torch.Tensor, target: torch.Tensor, dem: torch.Tensor,
                            water_class: int = 1, ignore_index: int = -1) -> dict[str, Any]:
    probs = torch.softmax(logits, dim=1)
    p_water = probs[:, water_class]
    pred = torch.argmax(logits, dim=1)
    valid = (target != ignore_index)
    n_valid = int(valid.sum().item())

    pw_valid = p_water[valid]
    pred_valid = pred[valid]
    tgt_valid = target[valid]

    n_pred_water = int((pred_valid == water_class).sum().item())
    n_pred_bg = int((pred_valid != water_class).sum().item())
    n_true_water = int((tgt_valid == water_class).sum().item())

    conf = t6c.confusion(target, pred)
    m = t6c.metrics_from_conf(conf)

    # class histograms (valid pixels only)
    pred_hist = [int((pred_valid == c).sum().item()) for c in range(logits.shape[1])]
    tgt_hist = [int((tgt_valid == c).sum().item()) for c in range(logits.shape[1])]

    return {
        "n_valid_pixels": n_valid,
        "true_water_fraction": (n_true_water / n_valid) if n_valid else None,
        "pred_water_fraction_argmax": (n_pred_water / n_valid) if n_valid else None,
        "pred_background_fraction_argmax": (n_pred_bg / n_valid) if n_valid else None,
        "p_water_mean": float(pw_valid.mean().item()) if n_valid else None,
        "p_water_median": float(pw_valid.median().item()) if n_valid else None,
        "p_water_min": float(pw_valid.min().item()) if n_valid else None,
        "p_water_max": float(pw_valid.max().item()) if n_valid else None,
        "pred_class_histogram": pred_hist,
        "target_class_histogram": tgt_hist,
        "iou_background": m["iou_background"],
        "iou_water": m["iou_water"],
        "mean_iou": m["mean_iou"],
        "f1_water": m["f1_water"],
        "confusion_tn_fp_fn_tp": [m["tn"], m["fp"], m["fn"], m["tp"]],
    }


def grad_l2(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {
        "step": "6C-lambda05-collapse-diagnostics",
        "device": str(device),
        "config_path": str(CONFIG_PATH),
        "collapsed_checkpoint": str(LAST_CKPT),
        "notes": [
            "Read-only diagnostic. No training, no raw-data change, DEM loss-only.",
        ],
    }

    with CONFIG_PATH.open("r", encoding="utf-8-sig") as fh:
        config = yaml.safe_load(fh)
    physics = config["physics_loss"]
    ignore_index = int(physics["ignore_index"])
    water_class = int(physics["water_class"])

    # ---- one real train batch + one real val batch (no D4: deterministic) ----
    dm = t6c.build_datamodule(config, batch_size=2, train_aug=True)  # train_aug=True => D4 manual, dataloader gives ToTensor only
    dm.setup("fit")
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))
    train_dem = t6c.load_dem_batch(config, train_batch, split="train")
    val_dem = t6c.load_dem_batch(config, val_batch, split="valid")

    # ---- build task (fresh pretrained init, identical to both runners) ----
    task = t6c.build_task(config).to(device)
    t6c.set_bn_eval(task)
    task.eval()

    n_total = sum(p.numel() for p in task.parameters())
    n_trainable = sum(p.numel() for p in task.parameters() if p.requires_grad)
    report["parameters"] = {"total": int(n_total), "trainable": int(n_trainable)}

    def forward(batch):
        b = t6c.move_batch(batch, device)
        out = task(b["image"])
        logits = t6c.get_logits(out)
        target = task.squeeze_ground_truth(b["mask"]).long()
        return out, logits, target

    # ===================== 7.3 prediction distribution =====================
    with torch.no_grad():
        _, ltr, ttr = forward(train_batch)
        _, lval, tval = forward(val_batch)
        report["prediction_distribution_fresh_init"] = {
            "train": prediction_distribution(ltr, ttr, train_dem.to(device), water_class, ignore_index),
            "valid": prediction_distribution(lval, tval, val_dem.to(device), water_class, ignore_index),
        }

    # ===================== 7.4 loss parity (fresh-init logits) =====================
    # terratorch path (EXACT 5S-A loss): task.train_loss_handler + task.criterion (smp DiceLoss)
    def parity_on(batch, dem, split_name):
        out, logits, target = forward(batch)
        with torch.no_grad():
            # 1. terratorch loss exactly as STEP 5S-A computes it
            tt = task.train_loss_handler.compute_loss(out, target, task.criterion, task.aux_loss)["loss"]
            # 2. smp DiceLoss constructed identically to terratorch's factory
            smp_dice = smp.losses.DiceLoss("multiclass", ignore_index=ignore_index)(logits, target)
            # 3. current hand-rolled _soft_dice_loss (what the FAILED 6C used)
            hand = _soft_dice_loss(logits, target, ignore_index=ignore_index, smooth=1.0)
            # 4. CombinedDicePhysicsLoss lambda=0 (current/old impl) -> loss_dice
            comb0 = CombinedDicePhysicsLoss(lambda_topo=0.0, ignore_index=ignore_index, water_class=water_class)
            comb0_out = comb0(logits=logits, target=target, topography=dem.to(device))
            # 5. CombinedDicePhysicsLoss lambda=0.5
            comb05 = CombinedDicePhysicsLoss(lambda_topo=0.5, ignore_index=ignore_index, water_class=water_class)
            comb05_out = comb05(logits=logits, target=target, topography=dem.to(device))
        return {
            "split": split_name,
            "terratorch_dice_5sa": float(tt.item()),
            "smp_dice_direct": float(smp_dice.item()),
            "handrolled_soft_dice_smooth1": float(hand.item()),
            "combined_lambda0_loss_dice": float(comb0_out["loss_dice"].item()),
            "combined_lambda0_loss_total": float(comb0_out["loss_total"].item()),
            "combined_lambda05_loss_dice": float(comb05_out["loss_dice"].item()),
            "combined_lambda05_loss_topo": float(comb05_out["loss_topo"].item()),
            "combined_lambda05_loss_total": float(comb05_out["loss_total"].item()),
            "abs_diff_combined0_vs_terratorch": abs(float(comb0_out["loss_dice"].item()) - float(tt.item())),
            "abs_diff_handrolled_vs_terratorch": abs(float(hand.item()) - float(tt.item())),
            "abs_diff_smp_vs_terratorch": abs(float(smp_dice.item()) - float(tt.item())),
        }

    report["loss_parity_fresh_init"] = {
        "train": parity_on(train_batch, train_dem, "train"),
        "valid": parity_on(val_batch, val_dem, "valid"),
    }

    # ===================== load collapsed checkpoint =====================
    collapsed_loaded = False
    if LAST_CKPT.exists():
        ckpt = torch.load(LAST_CKPT, map_location=device, weights_only=False)
        task.load_state_dict(ckpt["model_state_dict"])
        t6c.set_bn_eval(task)
        task.eval()
        collapsed_loaded = True
        report["collapsed_checkpoint_meta"] = {
            "epoch": int(ckpt.get("epoch", -1)),
            "best_validation_miou": ckpt.get("best_validation_miou"),
            "best_epoch": ckpt.get("best_epoch"),
        }

    if collapsed_loaded:
        with torch.no_grad():
            _, ltr, ttr = forward(train_batch)
            _, lval, tval = forward(val_batch)
            report["prediction_distribution_collapsed_ckpt"] = {
                "train": prediction_distribution(ltr, ttr, train_dem.to(device), water_class, ignore_index),
                "valid": prediction_distribution(lval, tval, val_dem.to(device), water_class, ignore_index),
            }
        report["loss_parity_collapsed_ckpt"] = {
            "train": parity_on(train_batch, train_dem, "train"),
        }

        # ===================== 7.6 gradient / optimizer diagnostics =====================
        # Use the collapsed model on a real train batch.
        task.train()
        t6c.set_bn_eval(task)
        params = [p for p in task.parameters() if p.requires_grad]

        out, logits, target = forward(train_batch)
        dem_t = train_dem.to(device)

        dice_only = CombinedDicePhysicsLoss(lambda_topo=0.0, ignore_index=ignore_index, water_class=water_class)
        topo_only = TopographicInconsistencyLoss(ignore_index=ignore_index, water_class=water_class,
                                                 elevation_margin=float(physics["elevation_margin"]),
                                                 elevation_scale=float(physics["elevation_scale"]),
                                                 use_elevation_weight=bool(physics["use_elevation_weight"]),
                                                 neighborhood=str(physics["neighborhood"]))
        combined = CombinedDicePhysicsLoss(lambda_topo=0.5, ignore_index=ignore_index, water_class=water_class)

        # grad norm: dice only
        task.zero_grad(set_to_none=True)
        d = dice_only(logits=logits, target=target, topography=dem_t)["loss_dice"]
        d.backward(retain_graph=True)
        g_dice = grad_l2(params)

        # grad norm: topo only
        task.zero_grad(set_to_none=True)
        tp = topo_only(logits=logits, target=target, topography=dem_t)
        tp.backward(retain_graph=True)
        g_topo = grad_l2(params)

        # grad norm: total
        task.zero_grad(set_to_none=True)
        tot = combined(logits=logits, target=target, topography=dem_t)["loss_total"]
        tot.backward()
        g_total = grad_l2(params)

        # one controlled optimizer step -> param delta
        before = torch.cat([p.detach().reshape(-1) for p in params]).clone()
        opt = torch.optim.AdamW(params, lr=float(config["optimizer"]["init_args"]["lr"]),
                                weight_decay=float(config["optimizer"]["init_args"]["weight_decay"]))
        # recompute fresh grads for the step (total loss)
        task.zero_grad(set_to_none=True)
        out2, logits2, target2 = forward(train_batch)
        tot2 = combined(logits=logits2, target=target2, topography=dem_t)["loss_total"]
        tot2.backward()
        opt.step()
        after = torch.cat([p.detach().reshape(-1) for p in params]).clone()
        param_delta = float((after - before).norm().item())

        report["gradient_diagnostics_collapsed"] = {
            "grad_l2_dice_only": g_dice,
            "grad_l2_topo_only": g_topo,
            "grad_l2_total": g_total,
            "topo_to_dice_grad_ratio": (g_topo / g_dice) if g_dice > 0 else None,
            "param_l2_delta_after_one_step": param_delta,
            "model_updates": param_delta > 0,
            "interpretation": "topo grad ~0 and tiny param delta => saturated/stuck collapse",
        }

    # ===================== 7.7 topo degeneracy probe (synthetic p_water=0.5) =====================
    # At an UN-saturated state, show topo gradient direction on the water logit channel:
    # does it suppress water on high pixels and/or encourage on low pixels?
    H = W = 128
    dem_small = val_dem[:, :H, :W].to(device).float()
    tgt_small = torch.zeros((dem_small.shape[0], H, W), dtype=torch.long, device=device)  # all valid, all bg
    logits_syn = torch.zeros((dem_small.shape[0], 2, H, W), device=device, requires_grad=True)  # p_water=0.5
    topo_probe = TopographicInconsistencyLoss(ignore_index=ignore_index, water_class=water_class,
                                              elevation_margin=0.0, elevation_scale=1.0,
                                              use_elevation_weight=True, neighborhood="4")
    lt = topo_probe(logits=logits_syn, target=tgt_small, topography=dem_small)
    lt.backward()
    g_water = logits_syn.grad[:, water_class]  # [B,H,W]; GD moves logit by -g

    # classify pixels as "high" vs "low" relative to 4-neighbour mean elevation
    dem_pad = torch.nn.functional.pad(dem_small.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    neigh_mean = (dem_pad[:, :, :-2, 1:-1] + dem_pad[:, :, 2:, 1:-1]
                  + dem_pad[:, :, 1:-1, :-2] + dem_pad[:, :, 1:-1, 2:]) / 4.0
    neigh_mean = neigh_mean[:, 0]
    is_high = dem_small > neigh_mean
    is_low = dem_small < neigh_mean

    gh = g_water[is_high]
    gl = g_water[is_low]
    report["topo_degeneracy_probe_synthetic_pwater_0p5"] = {
        "synthetic_logits": "zeros => p_water=0.5 everywhere (unsaturated)",
        "topo_loss_value": float(lt.item()),
        "mean_grad_water_logit_high_pixels": float(gh.mean().item()) if gh.numel() else None,
        "mean_grad_water_logit_low_pixels": float(gl.mean().item()) if gl.numel() else None,
        "frac_high_pixels_grad_positive_suppress_water": float((gh > 0).float().mean().item()) if gh.numel() else None,
        "frac_low_pixels_grad_negative_encourage_water": float((gl < 0).float().mean().item()) if gl.numel() else None,
        "note": "grad>0 on water logit => gradient descent SUPPRESSES water at that pixel.",
    }
    report["degenerate_minima"] = {
        "all_dry_p_water_0": "L_topo = 0 (p_high=0). Coincides with Dice majority-class (all-background) basin.",
        "all_wet_p_water_1": "L_topo = 0 (1-p_low=0). Opposed by Dice (high loss).",
        "physically_consistent": "L_topo = 0 for monotonic fills. The intended minimum.",
        "conclusion": "all-dry is a shared minimum of topo AND the early Dice basin => reinforced collapse.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(jsafe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(jsafe(report), indent=2, ensure_ascii=False))
    print(f"\nWROTE: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
