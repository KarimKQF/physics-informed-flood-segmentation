"""
STEP 6C v2 readiness smoke test (NO full training, NO run-artifact creation).

For each PRIMARY config (lambda=0.1 constant, lambda=0.5 warmup):
  - verify the lambda schedule values,
  - load ONE real train batch + aligned DEM,
  - forward -> Dice + topo + total, backward, check finite gradients,
  - report predicted water fraction before and after ONE controlled optimizer step.

For the SECONDARY finetune config:
  - verify the 5S-A best checkpoint warm-start loads (strict) and predicts a
    non-degenerate (non-zero) water fraction.

Output JSON:
  E:/flood_research/experiments/terramind_baseline/runs/step6c_v2_readiness_smoke/metrics/smoke_results.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import step6c_lambda05_train as t6c  # noqa: E402
import step6c_v2_train as v2  # noqa: E402

CONFIGS = {
    "lambda01": REPO_ROOT / "configs" / "step6c_v2_terramind_l_upernet_dice_topographic_lambda01.yaml",
    "warmup": REPO_ROOT / "configs" / "step6c_v2_terramind_l_upernet_dice_topographic_lambda05_warmup.yaml",
    "secondary": REPO_ROOT / "configs" / "step6c_secondary_finetune_from_5s_a_best_lambda01.yaml",
}
OUT_JSON = Path("E:/flood_research/experiments/terramind_baseline/runs/step6c_v2_readiness_smoke/metrics/smoke_results.json")


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh)


def water_fraction(logits: torch.Tensor, target: torch.Tensor, water_class: int = 1) -> float:
    pred = torch.argmax(logits, dim=1)
    valid = target != -1
    n = int(valid.sum().item())
    return float((pred[valid] == water_class).sum().item()) / n if n else float("nan")


def jsafe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): jsafe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsafe(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def schedule_check() -> dict[str, Any]:
    warm_cfg = load_cfg(CONFIGS["warmup"])["physics_loss"]
    const_cfg = load_cfg(CONFIGS["lambda01"])["physics_loss"]

    warm = {e: v2.lambda_for_epoch(warm_cfg, e) for e in [1, 2, 5, 6, 7, 13, 20, 21, 40]}
    const = {e: v2.lambda_for_epoch(const_cfg, e) for e in [1, 5, 20, 80]}

    checks = {
        "warmup_epoch1_is_0": warm[1] == 0.0,
        "warmup_epoch5_is_0": warm[5] == 0.0,
        "warmup_epoch6_in_open_0_0p5": 0.0 < warm[6] < 0.5,
        "warmup_epoch13_in_open_0_0p5": 0.0 < warm[13] < 0.5,
        "warmup_epoch20_is_0p5": abs(warm[20] - 0.5) < 1e-9,
        "warmup_epoch21_is_0p5": abs(warm[21] - 0.5) < 1e-9,
        "warmup_epoch40_is_0p5": abs(warm[40] - 0.5) < 1e-9,
        "constant_all_0p1": all(abs(v - 0.1) < 1e-9 for v in const.values()),
    }
    return {"warmup_values": warm, "constant_values": const, "checks": checks,
            "all_passed": all(checks.values())}


def main() -> int:
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {"step": "6C-v2-readiness-smoke", "device": str(device)}

    # ---- schedule (no model needed) ----
    report["lambda_schedule"] = schedule_check()

    # ---- shared data + model (built once; architecture identical across configs) ----
    base_cfg = load_cfg(CONFIGS["lambda01"])
    dm = t6c.build_datamodule(base_cfg, batch_size=int(base_cfg["trainer"]["batch_size"]), train_aug=True)
    dm.setup("fit")
    raw_batch = next(iter(dm.train_dataloader()))
    dem_cpu = t6c.load_dem_batch(base_cfg, raw_batch, split="train")
    batch = t6c.move_batch(raw_batch, device)
    dem = dem_cpu.to(device)

    task = t6c.build_task(base_cfg).to(device)
    t6c.set_bn_eval(task)

    def forward():
        out = task(batch["image"])
        logits = t6c.get_logits(out)
        target = task.squeeze_ground_truth(batch["mask"]).long()
        return logits, target

    def smoke_one(name: str, cfg_path: Path, test_epochs: list[int]) -> dict[str, Any]:
        cfg = load_cfg(cfg_path)
        physics = cfg["physics_loss"]
        criterion = v2.build_loss(cfg).to(device)
        res: dict[str, Any] = {"config": str(cfg_path), "epochs": {}}
        for ep in test_epochs:
            lam = v2.lambda_for_epoch(physics, ep)
            criterion.set_lambda_topo(lam)
            task.train(); t6c.set_bn_eval(task)
            logits, target = forward()
            wf_before = water_fraction(logits, target)
            losses = criterion(logits=logits, target=target, topography=dem)
            (losses["loss_total"]).backward()
            grads = [p.grad for p in task.parameters() if p.grad is not None]
            gfinite = all(torch.isfinite(g).all().item() for g in grads)
            gnorm = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))

            # one controlled optimizer step (throwaway optimizer)
            opt = torch.optim.AdamW([p for p in task.parameters() if p.requires_grad],
                                    lr=float(cfg["optimizer"]["init_args"]["lr"]),
                                    weight_decay=float(cfg["optimizer"]["init_args"]["weight_decay"]))
            opt.step()
            task.zero_grad(set_to_none=True)
            with torch.no_grad():
                logits2, target2 = forward()
                wf_after = water_fraction(logits2, target2)

            res["epochs"][ep] = {
                "lambda_topo": lam,
                "loss_dice": float(losses["loss_dice"].item()),
                "loss_topo": float(losses["loss_topo"].item()),
                "loss_total": float(losses["loss_total"].item()),
                "total_equals_dice_when_lambda0": (abs(float(losses["loss_total"].item()) - float(losses["loss_dice"].item())) < 1e-9) if lam == 0.0 else None,
                "total_finite": bool(torch.isfinite(losses["loss_total"]).item()),
                "grads_finite": bool(gfinite),
                "grad_l2": gnorm,
                "pred_water_fraction_before_step": wf_before,
                "pred_water_fraction_after_step": wf_after,
            }
            task.zero_grad(set_to_none=True)
        return res

    report["smoke_lambda01"] = smoke_one("lambda01", CONFIGS["lambda01"], test_epochs=[1])
    report["smoke_warmup"] = smoke_one("warmup", CONFIGS["warmup"], test_epochs=[1, 20])

    # restore a clean model for the secondary warm-start check
    del task
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sec_cfg = load_cfg(CONFIGS["secondary"])
    task = t6c.build_task(sec_cfg).to(device)
    warm = v2.maybe_init_from_checkpoint(task, sec_cfg, device)
    t6c.set_bn_eval(task); task.eval()
    with torch.no_grad():
        out = task(batch["image"]); logits = t6c.get_logits(out)
        target = task.squeeze_ground_truth(batch["mask"]).long()
        wf = water_fraction(logits, target)
    report["smoke_secondary"] = {
        "config": str(CONFIGS["secondary"]),
        "warm_start_checkpoint": warm,
        "warm_start_loaded": warm is not None,
        "pred_water_fraction_from_5sa_init": wf,
        "non_degenerate": (wf is not None and wf > 0.01),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(jsafe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(jsafe(report), indent=2, ensure_ascii=False))
    print(f"\nWROTE: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
