from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from urban_runoff.losses.topographic_loss import (
    SegmentationWithTopographicLoss,
    build_loss_from_config,
)


def load_topographic_loss_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("Topographic loss config must be a mapping.")
    return data


def build_topographic_criterion(config: dict[str, Any]) -> SegmentationWithTopographicLoss:
    return build_loss_from_config(config)


def training_loop_integration_note() -> str:
    return (
        "Use criterion(logits=logits, target=target, dem=dem) and backpropagate "
        "out['loss']. TODO: update dataset to return DEM aligned with image and mask."
    )
