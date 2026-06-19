from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.utils import ensure_external_data_tree, load_local_paths  # noqa: E402

DEFAULT_DOI = "10.5281/zenodo.12748982"
ZENODO_RECORD_API = "https://zenodo.org/api/records/12748982"
ZENODO_PAGE = "https://zenodo.org/records/12748982"


def read_dataset_source(config_path: Path) -> str:
    if not config_path.exists():
        return f"https://doi.org/{DEFAULT_DOI}"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return (
        config.get("datasets", {})
        .get("sturm_flood", {})
        .get("source", f"https://doi.org/{DEFAULT_DOI}")
    )


def query_zenodo(timeout: float) -> dict[str, Any] | None:
    try:
        response = requests.get(ZENODO_RECORD_API, timeout=timeout)
    except requests.RequestException:
        return None
    if not response.ok:
        return None
    return response.json()


def format_size(size: Any) -> str:
    if not isinstance(size, int):
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a STURM-Flood download plan.")
    parser.add_argument("--config", type=Path, default=Path("configs/local_paths.yaml"))
    parser.add_argument("--datasets-config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    try:
        paths = load_local_paths(args.config)
        ensure_external_data_tree(paths)
        report_path = paths.logs / "sturm_flood_download_plan.txt"
        source = read_dataset_source(args.datasets_config)
        record = query_zenodo(args.timeout)

        with report_path.open("w", encoding="utf-8") as report:
            report.write("STURM-Flood download plan\n")
            report.write(f"Generated at: {datetime.now(UTC).isoformat()}\n")
            report.write(f"Configured source: {source}\n")
            report.write(f"Official DOI: {DEFAULT_DOI}\n")
            report.write(f"Official Zenodo page: {ZENODO_PAGE}\n\n")

            if record is None:
                report.write("Zenodo metadata could not be resolved automatically.\n")
                report.write(
                    "Recommendation: open the Zenodo page manually and download "
                    "a small subset first.\n"
                )
            else:
                report.write(f"Title: {record.get('title', 'unknown')}\n")
                files = record.get("files") or []
                report.write(f"Files discovered: {len(files)}\n")
                for file_info in files:
                    filename = file_info.get("key") or file_info.get("filename") or "unknown"
                    size = format_size(file_info.get("size"))
                    links = file_info.get("links") or {}
                    report.write(f"- {filename} ({size})\n")
                    if links.get("self") or links.get("download"):
                        report.write(f"  url: {links.get('self') or links.get('download')}\n")
                report.write(
                    "\nRecommendation: inspect file naming, then download a small "
                    "Sentinel-1 subset first.\n"
                )

            report.write("\nNo files were downloaded by this script.\n")
            report.write("Suggested local target: D:/urban_runoff_data/raw/STURM-Flood\n")

        print(f"[OK] STURM-Flood plan written to: {report_path}")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
