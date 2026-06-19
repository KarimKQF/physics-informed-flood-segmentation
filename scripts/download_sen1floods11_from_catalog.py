from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

DEFAULT_ROOT = Path("data/raw/Sen1Floods11")
CATALOG_ROOT = DEFAULT_ROOT / "v1.1" / "catalog"


@dataclass(frozen=True)
class Asset:
    url: str
    destination: Path


def _iter_item_jsons(catalog_root: Path, collections: set[str] | None) -> Iterator[Path]:
    for path in sorted(catalog_root.rglob("*.json")):
        if path.name in {"catalog.json", "collection.json"}:
            continue
        if collections and not any(part in collections for part in path.parts):
            continue
        yield path


def _asset_destination(root: Path, url: str) -> Path:
    parsed = urlparse(url)
    marker = "/sen1floods11/"
    if marker in parsed.path:
        relative = parsed.path.split(marker, maxsplit=1)[1]
    else:
        relative = Path(parsed.path).name
    return root / relative


def collect_assets(
    root: Path,
    catalog_root: Path,
    collections: set[str] | None,
) -> list[Asset]:
    assets: dict[str, Asset] = {}
    for item_path in _iter_item_jsons(catalog_root, collections):
        payload = json.loads(item_path.read_text(encoding="utf-8"))
        for asset_info in payload.get("assets", {}).values():
            url = asset_info.get("href", "")
            asset_type = asset_info.get("type", "")
            if not url.lower().endswith((".tif", ".tiff")) and "geotiff" not in asset_type:
                continue
            destination = _asset_destination(root, url)
            assets[url] = Asset(url=url, destination=destination)
    return list(assets.values())


def _download(asset: Asset, timeout: float) -> str:
    asset.destination.parent.mkdir(parents=True, exist_ok=True)
    if asset.destination.exists() and asset.destination.stat().st_size > 0:
        return "exists"

    temporary = asset.destination.with_suffix(asset.destination.suffix + ".part")
    with requests.get(asset.url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        with temporary.open("wb") as file:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=asset.destination.name,
                leave=False,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file.write(chunk)
                    progress.update(len(chunk))
    temporary.replace(asset.destination)
    return "downloaded"


def download_assets(
    assets: list[Asset],
    *,
    dry_run: bool,
    limit: int | None,
    timeout: float,
) -> dict[str, int]:
    selected = assets[:limit] if limit is not None else assets
    counts = {"planned": len(selected), "downloaded": 0, "exists": 0, "failed": 0}

    for asset in selected:
        if dry_run:
            print(f"DRY RUN {asset.url} -> {asset.destination}")
            continue
        try:
            status = _download(asset, timeout)
            counts[status] += 1
        except Exception as exc:  # noqa: BLE001
            counts["failed"] += 1
            print(f"FAILED {asset.url}: {exc}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Sen1Floods11 GeoTIFF assets from local STAC catalog JSON files."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--catalog-root", type=Path, default=CATALOG_ROOT)
    parser.add_argument(
        "--collection",
        action="append",
        choices=[
            "sen1floods11_hand_labeled_source",
            "sen1floods11_hand_labeled_label",
            "sen1floods11_weak_labeled_source",
            "sen1floods11_weak_labeled_label",
        ],
        help=(
            "Collection to download. Can be passed multiple times. Defaults to all local catalogs."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    if not args.catalog_root.exists():
        raise SystemExit(
            f"Catalog root not found: {args.catalog_root}. "
            "Run the metadata download first or copy the STAC catalog locally."
        )

    collections = set(args.collection) if args.collection else None
    assets = collect_assets(args.root, args.catalog_root, collections)
    print(f"Catalog root: {args.catalog_root}")
    print(f"Assets discovered: {len(assets)}")
    if args.limit is not None:
        print(f"Limit: {args.limit}")

    counts = download_assets(
        assets,
        dry_run=args.dry_run,
        limit=args.limit,
        timeout=args.timeout,
    )
    print(f"Summary: {counts}")


if __name__ == "__main__":
    main()
