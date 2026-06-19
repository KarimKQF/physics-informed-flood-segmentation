from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

DOI = "10.5281/zenodo.12748982"
ZENODO_URL = "https://zenodo.org/records/12748982"
ZENODO_API = "https://zenodo.org/api/records"
ZENODO_RECORD_API = "https://zenodo.org/api/records/12748982"


def _query_zenodo_record(timeout: float) -> dict[str, Any] | None:
    try:
        response = requests.get(ZENODO_RECORD_API, timeout=timeout)
        if response.ok:
            return response.json()
    except requests.RequestException:
        pass

    queries = [f'doi:"{DOI}"', f'conceptdoi:"{DOI}"']
    for query in queries:
        try:
            response = requests.get(
                ZENODO_API,
                params={"q": query, "size": 1},
                timeout=timeout,
            )
        except requests.RequestException:
            continue
        if not response.ok:
            continue
        payload = response.json()
        hits = payload.get("hits", {}).get("hits", [])
        if hits:
            return hits[0]
    return None


def _manual_instructions(output_dir: Path) -> None:
    print("Could not discover downloadable files automatically through the Zenodo API.")
    print(f"DOI: {DOI}")
    print(f"Zenodo page: {ZENODO_URL}")
    print(f"Place downloaded files manually in: {output_dir}")
    print("Then run: python scripts/inspect_sturm_flood.py")


def _download_file(url: str, destination: Path, expected_size: int | None) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or expected_size or 0)
        with destination.open("wb") as file:
            with tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file.write(chunk)
                    progress.update(len(chunk))


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to: {manifest_path}")


def download_sturm_flood(
    output_dir: Path,
    dry_run: bool,
    timeout: float,
    resolve_metadata: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run and not resolve_metadata:
        manifest = {
            "doi": DOI,
            "zenodo_url": ZENODO_URL,
            "generated_at": datetime.now(UTC).isoformat(),
            "dry_run": True,
            "metadata_resolved": False,
            "files": [],
        }
        print("Dry run only: no files will be downloaded and Zenodo is not queried.")
        _manual_instructions(output_dir)
        print("To query Zenodo metadata during dry-run, add: --resolve-metadata")
        _write_manifest(output_dir, manifest)
        return

    record = _query_zenodo_record(timeout)
    if record is None:
        _manual_instructions(output_dir)
        return

    files = record.get("files") or []
    print(f"Zenodo record: {record.get('title', 'unknown title')}")
    print(f"DOI: {DOI}")
    print(f"Output directory: {output_dir}")
    print(f"Files discovered: {len(files)}")

    manifest: dict[str, Any] = {
        "doi": DOI,
        "zenodo_url": ZENODO_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "files": [],
    }

    for file_info in files:
        filename = file_info.get("key") or file_info.get("filename")
        links = file_info.get("links") or {}
        url = links.get("self") or links.get("download")
        size = file_info.get("size")
        if not filename or not url:
            continue

        destination = output_dir / filename
        status = "planned"
        size_label = f"{size / (1024**2):.2f} MiB" if isinstance(size, int) else "unknown size"

        if destination.exists():
            status = "exists"
            print(f"Skipping existing file: {destination.name} ({size_label})")
        elif dry_run:
            print(f"Would download: {destination.name} ({size_label})")
        else:
            print(f"Downloading: {destination.name} ({size_label})")
            _download_file(url, destination, size if isinstance(size, int) else None)
            status = "downloaded"

        manifest["files"].append({"filename": filename, "url": url, "size": size, "status": status})

    _write_manifest(output_dir, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download STURM-Flood files from Zenodo.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/STURM-Flood"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resolve-metadata",
        action="store_true",
        help="Query Zenodo during dry-run to list files and sizes when network is available.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Network timeout in seconds for Zenodo metadata requests.",
    )
    args = parser.parse_args()
    download_sturm_flood(args.output_dir, args.dry_run, args.timeout, args.resolve_metadata)


if __name__ == "__main__":
    main()
