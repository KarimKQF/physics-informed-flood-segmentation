from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LocalDataPaths:
    data_root: Path
    raw_sen1floods11: Path
    raw_sturm_flood: Path
    raw_dem: Path
    processed_sen1floods11: Path
    processed_sturm_flood: Path
    processed_aligned_dem: Path
    manifests: Path
    logs: Path
    tmp: Path

    def all_directories(self) -> dict[str, Path]:
        return {
            "data_root": self.data_root,
            "raw/Sen1Floods11": self.raw_sen1floods11,
            "raw/STURM-Flood": self.raw_sturm_flood,
            "raw/DEM": self.raw_dem,
            "processed/Sen1Floods11": self.processed_sen1floods11,
            "processed/STURM-Flood": self.processed_sturm_flood,
            "processed/aligned_dem": self.processed_aligned_dem,
            "manifests": self.manifests,
            "logs": self.logs,
            "tmp": self.tmp,
        }


def _path_from(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            joined = ".".join(keys)
            raise KeyError(f"Missing path config key: {joined}")
        value = value[key]
    return Path(str(value))


def load_local_paths(config_path: str | Path) -> LocalDataPaths:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Local paths config does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Local paths config must be a mapping: {path}")

    return LocalDataPaths(
        data_root=_path_from(config, "data_root"),
        raw_sen1floods11=_path_from(config, "raw", "sen1floods11"),
        raw_sturm_flood=_path_from(config, "raw", "sturm_flood"),
        raw_dem=_path_from(config, "raw", "dem"),
        processed_sen1floods11=_path_from(config, "processed", "sen1floods11"),
        processed_sturm_flood=_path_from(config, "processed", "sturm_flood"),
        processed_aligned_dem=_path_from(config, "processed", "aligned_dem"),
        manifests=_path_from(config, "manifests"),
        logs=_path_from(config, "logs"),
        tmp=_path_from(config, "tmp"),
    )


def ensure_external_data_tree(paths: LocalDataPaths) -> None:
    for directory in paths.all_directories().values():
        directory.mkdir(parents=True, exist_ok=True)
