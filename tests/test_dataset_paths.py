from pathlib import Path

import yaml


def test_dataset_config_exists_and_contains_expected_keys() -> None:
    config_path = Path("configs/datasets.yaml")
    assert config_path.exists()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "datasets" in config
    assert "sen1floods11" in config["datasets"]
    assert "sturm_flood" in config["datasets"]
    assert config["datasets"]["sen1floods11"]["root"] == "data/raw/Sen1Floods11"
    assert config["datasets"]["sturm_flood"]["root"] == "data/raw/STURM-Flood"


def test_dataset_scripts_exist() -> None:
    expected_scripts = [
        "scripts/download_sen1floods11.sh",
        "scripts/download_sen1floods11_from_catalog.py",
        "scripts/download_sturm_flood.py",
        "scripts/inspect_sen1floods11.py",
        "scripts/inspect_sturm_flood.py",
    ]

    for script in expected_scripts:
        assert Path(script).exists()


def test_dataset_documentation_exists() -> None:
    assert Path("README_datasets.md").exists()
    assert Path("data/README.md").exists()
