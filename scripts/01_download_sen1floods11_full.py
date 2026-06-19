"""STEP 1: full Sen1Floods11 download.

Downloads the complete public Sen1Floods11 bucket into the validated external
storage root and writes a post-download report. This script stops after the
download/verification step; it does not index, audit, clean, or train models.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


SOURCE_GCS_URI = "gs://sen1floods11"
SOURCE_REPO_URL = "https://github.com/cloudtostreet/Sen1Floods11"
EXPECTED_REMOTE_BYTES = 36_864_294_387
EXPECTED_REMOTE_OBJECTS = 31_074
SAFETY_RESERVE_BYTES = 50 * 1_000_000_000

KEY_PATHS = {
    "HandLabeled": Path("v1.1/data/flood_events/HandLabeled"),
    "WeaklyLabeled": Path("v1.1/data/flood_events/WeaklyLabeled"),
    "perm_water": Path("v1.1/data/perm_water"),
    "S1Perm": Path("v1.1/data/perm_water/S1Perm"),
    "JRCPerm": Path("v1.1/data/perm_water/JRCPerm"),
    "splits": Path("v1.1/splits"),
    "flood_handlabeled_splits": Path("v1.1/splits/flood_handlabeled"),
    "perm_water_splits": Path("v1.1/splits/perm_water"),
    "catalog": Path("v1.1/catalog"),
    "catalog_zip": Path("v1.1/catalog.zip"),
    "metadata": Path("v1.1/Sen1Floods11_Metadata.geojson"),
    "checkpoints": Path("v1.1/checkpoints"),
}


@dataclass(frozen=True)
class LocalTreeStats:
    file_count: int
    total_bytes: int


def bytes_to_gb(value: int | float) -> float:
    return round(float(value) / 1_000_000_000, 3)


def bytes_to_gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


def format_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config: {path}")
    return payload


def find_gsutil() -> str:
    candidates = [
        shutil.which("gsutil"),
        shutil.which("gsutil.cmd"),
        r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd",
        r"C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("gsutil was not found. Install Google Cloud SDK and retry.")


def tree_stats(root: Path) -> LocalTreeStats:
    file_count = 0
    total_bytes = 0
    if not root.exists():
        return LocalTreeStats(file_count=0, total_bytes=0)
    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return LocalTreeStats(file_count=file_count, total_bytes=total_bytes)


def key_prefix_presence(root: Path) -> dict[str, bool]:
    return {name: (root / relative).exists() for name, relative in KEY_PATHS.items()}


def count_key_files(root: Path) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for name, relative in KEY_PATHS.items():
        path = root / relative
        if path.is_file():
            counts[name] = 1
        elif path.is_dir():
            counts[name] = sum(1 for child in path.rglob("*") if child.is_file())
        else:
            counts[name] = None
    return counts


def run_rsync(gsutil: str, target: Path, log_path: Path) -> int:
    command = [gsutil, "-m", "rsync", "-r", SOURCE_GCS_URI, str(target)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(f"{datetime.now().isoformat(timespec='seconds')} STEP 1 download start\n")
        log.write(f"Command: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
        log.write(f"{datetime.now().isoformat(timespec='seconds')} return_code={return_code}\n")
        log.flush()
    return return_code


def build_report(
    *,
    generated_at: str,
    status: str,
    repo_root: Path,
    config_path: Path,
    target: Path,
    reports_dir: Path,
    logs_dir: Path,
    log_path: Path,
    local_report: Path,
    external_report: Path,
    local_status: Path,
    gsutil: str,
    command: list[str],
    return_code: int | None,
    stats: LocalTreeStats,
    key_present: dict[str, bool],
    key_counts: dict[str, int | None],
    free_bytes: int,
    total_bytes_disk: int,
    blocking_reasons: list[str],
    skipped_or_failed_hint: str,
) -> str:
    remaining_after_expected = free_bytes
    all_keys_present = all(key_present.values())
    size_delta = stats.total_bytes - EXPECTED_REMOTE_BYTES
    count_delta = stats.file_count - EXPECTED_REMOTE_OBJECTS

    lines = [
        "# STEP 1 - Full Sen1Floods11 download report",
        "",
        "## Summary",
        f"- Status: `{status}`",
        f"- Generated at: `{generated_at}`",
        "- Full download requested: `true`",
        "- STEP 2 started: `false`",
        "- Next step allowed: `false`",
        "",
        "## Source",
        f"- Official repository: {SOURCE_REPO_URL}",
        f"- GCS bucket: `{SOURCE_GCS_URI}`",
        f"- Download command: `{' '.join(command)}`",
        f"- gsutil executable: `{gsutil}`",
        f"- gsutil return code: `{return_code}`",
        "",
        "## Storage",
        f"- Local repo root: `{format_path(repo_root)}`",
        f"- Config file: `{format_path(config_path)}`",
        f"- Download target: `{format_path(target)}`",
        f"- Reports directory: `{format_path(reports_dir)}`",
        f"- Logs directory: `{format_path(logs_dir)}`",
        f"- Download log: `{format_path(log_path)}`",
        f"- Disk total space: `{bytes_to_gb(total_bytes_disk)} GB` (`{bytes_to_gib(total_bytes_disk)} GiB`)",
        f"- Remaining free space: `{bytes_to_gb(free_bytes)} GB` (`{bytes_to_gib(free_bytes)} GiB`)",
        f"- Remaining free space after download: `{bytes_to_gb(remaining_after_expected)} GB`",
        "",
        "## Local dataset verification",
        f"- Local file count: `{stats.file_count}`",
        f"- Expected remote object count from STEP 1A: `{EXPECTED_REMOTE_OBJECTS}`",
        f"- File count delta: `{count_delta}`",
        f"- Local total size: `{bytes_to_gb(stats.total_bytes)} GB` (`{bytes_to_gib(stats.total_bytes)} GiB`)",
        f"- Expected remote size from STEP 1A: `{bytes_to_gb(EXPECTED_REMOTE_BYTES)} GB` (`{bytes_to_gib(EXPECTED_REMOTE_BYTES)} GiB`)",
        f"- Size delta: `{bytes_to_gb(size_delta)} GB`",
        f"- Key prefixes present: `{str(all_keys_present).lower()}`",
        "",
        "## Key prefixes",
        "| Key | Present | File count | Local path |",
        "|---|---:|---:|---|",
    ]
    for name, relative in KEY_PATHS.items():
        count = key_counts[name]
        count_text = "N/A" if count is None else str(count)
        lines.append(
            f"| `{name}` | `{str(key_present[name]).lower()}` | {count_text} | "
            f"`{format_path(target / relative)}` |"
        )

    lines.extend(
        [
            "",
            "## Failed or skipped objects",
            f"- gsutil return code: `{return_code}`",
            f"- Log-based hint: {skipped_or_failed_hint}",
            "- See the full log for object-level transfer details.",
            "",
            "## Problems detected",
        ]
    )
    if blocking_reasons:
        lines.extend(f"- {reason}" for reason in blocking_reasons)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Generated files",
            f"- Local report: `{format_path(local_report)}`",
            f"- External report: `{format_path(external_report)}`",
            f"- Pipeline status: `{format_path(local_status)}`",
            "",
            "## Decision",
            "- Stop after STEP 1.",
            "- Do not start STEP 2 until human validation.",
            "- Do not index, audit, clean, or train models yet.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def summarize_log(log_path: Path) -> str:
    if not log_path.exists():
        return "download log not found"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    failure_tokens = ["exception", "traceback", "commandexception", "error", "failed"]
    skip_tokens = ["skip", "skipping"]
    failures = sum(lowered.count(token) for token in failure_tokens)
    skips = sum(lowered.count(token) for token in skip_tokens)
    return f"`{failures}` failure/error keyword hits, `{skips}` skip keyword hits"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sen1floods11.yaml"))
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only verify existing local data and write reports.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    target = Path(str(config["raw"]["sen1floods11"]))  # type: ignore[index]
    reports_dir = Path(str(config["reports"]))
    logs_dir = Path(str(config["logs"]))
    external_project_root = Path(str(config["project_root"]))

    target.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    external_project_root.mkdir(parents=True, exist_ok=True)
    (external_project_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (external_project_root / "data" / "processed" / "sen1floods11").mkdir(
        parents=True, exist_ok=True
    )

    gsutil = find_gsutil()
    command = [gsutil, "-m", "rsync", "-r", SOURCE_GCS_URI, str(target)]
    log_path = logs_dir / "01_download_sen1floods11_full.log"
    generated_at = datetime.now().isoformat(timespec="seconds")
    return_code: int | None = None

    if not args.skip_download:
        return_code = run_rsync(gsutil, target, log_path)

    stats = tree_stats(target)
    key_present = key_prefix_presence(target)
    key_counts = count_key_files(target)
    disk_usage = shutil.disk_usage(target.anchor or target)
    skipped_or_failed_hint = summarize_log(log_path)

    blocking_reasons: list[str] = []
    if return_code not in {0, None}:
        blocking_reasons.append(f"gsutil rsync returned non-zero code {return_code}.")
    if not all(key_present.values()):
        missing = [name for name, present in key_present.items() if not present]
        blocking_reasons.append(f"Missing expected key paths: {', '.join(missing)}.")
    if stats.total_bytes < EXPECTED_REMOTE_BYTES * 0.99:
        blocking_reasons.append("Local total size is below 99% of the STEP 1A remote estimate.")
    if stats.file_count < EXPECTED_REMOTE_OBJECTS * 0.99:
        blocking_reasons.append("Local file count is below 99% of the STEP 1A remote object count.")
    if disk_usage.free < SAFETY_RESERVE_BYTES:
        blocking_reasons.append("Remaining free space is below the 50 GB safety reserve.")

    status = "blocked" if blocking_reasons else "done"

    local_report = repo_root / "reports" / "STEP_1_full_download_report.md"
    external_report = reports_dir / "STEP_1_full_download_report.md"
    local_status = repo_root / "pipeline_status.json"
    external_status = external_project_root / "pipeline_status.json"

    report = build_report(
        generated_at=generated_at,
        status=status,
        repo_root=repo_root,
        config_path=config_path,
        target=target,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
        log_path=log_path,
        local_report=local_report,
        external_report=external_report,
        local_status=local_status,
        gsutil=gsutil,
        command=command,
        return_code=return_code,
        stats=stats,
        key_present=key_present,
        key_counts=key_counts,
        free_bytes=disk_usage.free,
        total_bytes_disk=disk_usage.total,
        blocking_reasons=blocking_reasons,
        skipped_or_failed_hint=skipped_or_failed_hint,
    )

    local_report.parent.mkdir(parents=True, exist_ok=True)
    local_report.write_text(report, encoding="utf-8")
    external_report.write_text(report, encoding="utf-8")

    status_payload: dict[str, object] = {
        "current_step": "1",
        "status": status,
        "full_download": True,
        "source_gcs_uri": SOURCE_GCS_URI,
        "external_disk_path": format_path(Path(str(config["external_disk_path"]))),
        "filesystem": "exFAT_or_NTFS_validated_in_STEP_0R",
        "download_target": format_path(target),
        "download_log": format_path(log_path),
        "gsutil_return_code": return_code,
        "total_downloaded_size_gb": bytes_to_gb(stats.total_bytes),
        "total_downloaded_size_gib": bytes_to_gib(stats.total_bytes),
        "total_file_count": stats.file_count,
        "remaining_free_space_gb": bytes_to_gb(disk_usage.free),
        "remaining_free_space_gib": bytes_to_gib(disk_usage.free),
        "key_prefixes_present": key_present,
        "key_file_counts": key_counts,
        "skipped_or_failed_hint": skipped_or_failed_hint,
        "blocking_reasons": blocking_reasons,
        "next_step_allowed": False,
        "human_validation_required": True,
        "generated_at": generated_at,
    }
    write_json(local_status, status_payload)
    write_json(external_status, status_payload)

    print("\nSTEP 1 - Full Sen1Floods11 download summary")
    print(f"Status: {status}")
    print(f"Target: {format_path(target)}")
    print(f"Total downloaded size: {bytes_to_gb(stats.total_bytes)} GB")
    print(f"Total file count: {stats.file_count}")
    print(f"Remaining free space: {bytes_to_gb(disk_usage.free)} GB")
    print(f"Key prefixes present: {all(key_present.values())}")
    print(f"gsutil return code: {return_code}")
    print("STOP: Human validation required before STEP 2.")

    return 0 if status == "done" else 2


if __name__ == "__main__":
    raise SystemExit(main())
