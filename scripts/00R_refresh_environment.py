"""STEP 0R: refresh environment and external storage validation.

This script validates the explicitly approved external root before any full
dataset download. The current workflow accepts D:/ as the validated storage
target when it is exFAT/NTFS, has roughly 2 TB total space, and has enough
free space for Sen1Floods11 plus reserve space.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml


LEGACY_LABEL = "ESD-USB"
DEFAULT_VALIDATED_ROOT = "D:/"
PROJECT_DIR_NAME = "flood_research"
ACCEPTABLE_FILESYSTEMS = {"EXFAT", "NTFS"}
OLD_PARTITION_LIMIT_BYTES = 40 * 1_000_000_000
EXPECTED_EXTERNAL_TOTAL_BYTES = 1_500 * 1_000_000_000
SEN1FLOODS11_ESTIMATED_BYTES = 36_864_294_387
SAFETY_RESERVE_BYTES = 50 * 1_000_000_000


@dataclass(frozen=True)
class VolumeInfo:
    root: str
    label: str
    filesystem: str
    total_bytes: int
    free_bytes: int
    drive_type: str = "unknown"


def bytes_to_gb(value: int | float) -> float:
    return round(float(value) / 1_000_000_000, 3)


def bytes_to_gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


def format_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def normalize_root(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    if len(value) == 2 and value[1] == ":":
        value += "/"
    if not value.endswith("/"):
        value += "/"
    return value.casefold()


def run_command(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": "command not found",
        }
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def find_gsutil() -> str | None:
    candidates = [
        shutil.which("gsutil"),
        shutil.which("gsutil.cmd"),
        r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd",
        r"C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def list_windows_volumes() -> list[VolumeInfo]:
    volumes: list[VolumeInfo] = []
    kernel32 = ctypes.windll.kernel32
    get_volume_information = kernel32.GetVolumeInformationW

    for code in range(ord("A"), ord("Z") + 1):
        root = f"{chr(code)}:\\"
        if not os.path.exists(root):
            continue

        volume_name = ctypes.create_unicode_buffer(261)
        fs_name = ctypes.create_unicode_buffer(261)
        serial_number = ctypes.c_ulong()
        max_component_len = ctypes.c_ulong()
        file_system_flags = ctypes.c_ulong()

        ok = get_volume_information(
            ctypes.c_wchar_p(root),
            volume_name,
            len(volume_name),
            ctypes.byref(serial_number),
            ctypes.byref(max_component_len),
            ctypes.byref(file_system_flags),
            fs_name,
            len(fs_name),
        )
        if not ok:
            continue

        usage = shutil.disk_usage(root)
        drive_type_code = kernel32.GetDriveTypeW(ctypes.c_wchar_p(root))
        drive_type = {
            0: "unknown",
            1: "no_root_dir",
            2: "removable",
            3: "fixed",
            4: "remote",
            5: "cdrom",
            6: "ramdisk",
        }.get(drive_type_code, "unknown")
        volumes.append(
            VolumeInfo(
                root=root,
                label=volume_name.value,
                filesystem=fs_name.value,
                total_bytes=usage.total,
                free_bytes=usage.free,
                drive_type=drive_type,
            )
        )
    return volumes


def list_unix_volumes() -> list[VolumeInfo]:
    candidates: list[Path] = []
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    if platform.system() == "Darwin":
        candidates.extend(Path("/Volumes").glob("*"))
    else:
        if user:
            candidates.extend((Path("/media") / user).glob("*"))
            candidates.extend((Path("/run/media") / user).glob("*"))
        candidates.extend(Path("/mnt").glob("*"))
        candidates.extend(Path("/media").glob("*"))

    volumes: list[VolumeInfo] = []
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        try:
            usage = shutil.disk_usage(candidate)
        except OSError:
            continue
        volumes.append(
            VolumeInfo(
                root=str(candidate),
                label=candidate.name,
                filesystem="unknown",
                total_bytes=usage.total,
                free_bytes=usage.free,
                drive_type="mount",
            )
        )
    return volumes


def list_volumes() -> list[VolumeInfo]:
    if platform.system() == "Windows":
        return list_windows_volumes()
    return list_unix_volumes()


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return payload if isinstance(payload, dict) else {}


def external_dirs(project_root: Path) -> list[Path]:
    return [
        project_root / "data" / "raw" / "sen1floods11",
        project_root / "data" / "processed" / "sen1floods11",
        project_root / "reports",
        project_root / "logs",
        project_root / "checkpoints",
    ]


def build_report(
    *,
    generated_at: str,
    repo_root: Path,
    config_path: Path,
    volumes: list[VolumeInfo],
    selected_volume: VolumeInfo | None,
    validated_root: Path,
    config_project_root: Path | None,
    status: str,
    blocking_reasons: list[str],
    created_or_verified_dirs: list[Path],
    generated_files: list[Path],
    python_info: str,
    pip_info: str,
    gsutil_info: str,
) -> str:
    lines = [
        "# STEP 0R - Environment refresh report",
        "",
        "## Summary",
        f"- Status: `{status}`",
        f"- Generated at: `{generated_at}`",
        "- Download launched: `false`",
        "- Next step allowed: `false`",
        f"- Local repo root: `{format_path(repo_root)}`",
        "",
        "## Environment",
        f"- Python: `{python_info}`",
        f"- pip: `{pip_info}`",
        f"- gsutil: `{gsutil_info}`",
        "",
        "## Required storage rules",
        f"- Legacy disk label: `{LEGACY_LABEL}`",
        "- Label requirement: `disabled by explicit human validation`",
        f"- Validated external root: `{format_path(validated_root)}`",
        "- Acceptable filesystems: `exFAT`, `NTFS`",
        f"- Minimum expected total capacity: `{bytes_to_gb(EXPECTED_EXTERNAL_TOTAL_BYTES)} GB`",
        f"- Estimated Sen1Floods11 full bucket size: `{bytes_to_gb(SEN1FLOODS11_ESTIMATED_BYTES)} GB`",
        f"- Required safety reserve: `{bytes_to_gb(SAFETY_RESERVE_BYTES)} GB`",
        "",
        "## Detected volumes",
        "| Root | Label | Filesystem | Total GB | Free GB | Drive type |",
        "|---|---|---|---:|---:|---|",
    ]
    for volume in volumes:
        lines.append(
            "| "
            f"`{format_path(volume.root)}` | "
            f"`{volume.label or '<empty>'}` | "
            f"`{volume.filesystem or 'unknown'}` | "
            f"{bytes_to_gb(volume.total_bytes)} | "
            f"{bytes_to_gb(volume.free_bytes)} | "
            f"`{volume.drive_type}` |"
        )

    lines.extend(["", "## Selected validated external volume"])
    if selected_volume is None:
        lines.append(f"- No mounted volume was detected at `{format_path(validated_root)}`.")
    else:
        free_after = selected_volume.free_bytes - SEN1FLOODS11_ESTIMATED_BYTES
        lines.extend(
            [
                f"- Disk path: `{format_path(selected_volume.root)}`",
                f"- Current label: `{selected_volume.label or '<empty>'}`",
                f"- Filesystem: `{selected_volume.filesystem}`",
                f"- Total space: `{bytes_to_gb(selected_volume.total_bytes)} GB`",
                f"- Free space: `{bytes_to_gb(selected_volume.free_bytes)} GB`",
                f"- Free space after estimated download: `{bytes_to_gb(free_after)} GB`",
                f"- Old 33 GB partition limitation cleared: `{str(selected_volume.total_bytes > OLD_PARTITION_LIMIT_BYTES).lower()}`",
                f"- Around 2 TB external capacity confirmed: `{str(selected_volume.total_bytes >= EXPECTED_EXTERNAL_TOTAL_BYTES).lower()}`",
            ]
        )

    lines.extend(["", "## Directories"])
    if created_or_verified_dirs:
        lines.extend(f"- `{format_path(path)}`" for path in created_or_verified_dirs)
    else:
        lines.append("- Not created because validation failed before storage acceptance.")

    lines.extend(["", "## Generated files"])
    lines.extend(f"- `{format_path(path)}`" for path in generated_files)

    lines.extend(["", "## Blocking reasons"])
    if blocking_reasons:
        lines.extend(f"- {reason}" for reason in blocking_reasons)
    else:
        lines.append("- None")

    lines.extend(["", "## Decision"])
    if status == "done":
        lines.extend(
            [
                "- Storage validation passed.",
                "- STEP 1 full download may proceed automatically in this run.",
            ]
        )
    else:
        lines.extend(
            [
                "- STEP 1 full download was not started.",
                "- Human action is required before continuing.",
            ]
        )

    lines.extend(
        [
            "",
            "## Proposed next step",
            "If this step is valid, continue to STEP 1 full download. If blocked, fix the listed storage issue and rerun STEP 0R.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sen1floods11.yaml"))
    parser.add_argument(
        "--validated-root",
        type=Path,
        default=None,
        help="Explicitly approved external storage root. Defaults to config external_disk_path or D:/.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    config_project_root = Path(str(config["project_root"])) if "project_root" in config else None
    validated_root = args.validated_root
    if validated_root is None and "external_disk_path" in config:
        validated_root = Path(str(config["external_disk_path"]))
    if validated_root is None:
        validated_root = Path(DEFAULT_VALIDATED_ROOT)

    generated_at = datetime.now().isoformat(timespec="seconds")
    volumes = list_volumes()
    selected = next(
        (volume for volume in volumes if normalize_root(volume.root) == normalize_root(validated_root)),
        None,
    )

    blocking_reasons: list[str] = []
    created_or_verified_dirs: list[Path] = []

    if selected is None:
        blocking_reasons.append(f"Validated external root {format_path(validated_root)!r} was not found.")
    else:
        filesystem = selected.filesystem.upper()
        if filesystem == "FAT32":
            blocking_reasons.append("Filesystem is FAT32, which is not acceptable for continuing.")
        elif filesystem not in ACCEPTABLE_FILESYSTEMS:
            blocking_reasons.append(
                f"Filesystem {selected.filesystem!r} is not in the accepted set exFAT/NTFS."
            )
        if selected.total_bytes <= OLD_PARTITION_LIMIT_BYTES:
            blocking_reasons.append("Disk still appears limited to the old ~33 GB partition.")
        if selected.total_bytes < EXPECTED_EXTERNAL_TOTAL_BYTES:
            blocking_reasons.append("Disk total capacity is below the expected external 2 TB class.")
        required_free = SEN1FLOODS11_ESTIMATED_BYTES + SAFETY_RESERVE_BYTES
        if selected.free_bytes < required_free:
            blocking_reasons.append(
                "Free space is insufficient for the full bucket plus 50 GB safety reserve."
            )

    status = "blocked" if blocking_reasons else "done"

    if selected is not None and status == "done":
        project_root = Path(selected.root) / PROJECT_DIR_NAME
        for directory in external_dirs(project_root):
            directory.mkdir(parents=True, exist_ok=True)
            created_or_verified_dirs.append(directory)
    elif config_project_root is not None:
        project_root = config_project_root
    else:
        project_root = None

    local_report = repo_root / "reports" / "STEP_0R_environment_refresh_report.md"
    local_status = repo_root / "pipeline_status.json"
    generated_files = [local_report, local_status]

    external_report = None
    external_status = None
    if project_root is not None and project_root.exists():
        external_report = project_root / "reports" / "STEP_0R_environment_refresh_report.md"
        external_status = project_root / "pipeline_status.json"
        generated_files.extend([external_report, external_status])

    python_info = f"{platform.python_version()} ({sys.executable})"
    pip_result = run_command([sys.executable, "-m", "pip", "--version"])
    pip_info = str(pip_result.get("stdout") or pip_result.get("stderr"))
    gsutil = find_gsutil()
    gsutil_result = run_command([gsutil, "version", "-l"]) if gsutil else {
        "stdout": "",
        "stderr": "command not found",
    }
    gsutil_info = str(gsutil_result.get("stdout") or gsutil_result.get("stderr")).splitlines()[0]

    report = build_report(
        generated_at=generated_at,
        repo_root=repo_root,
        config_path=config_path,
        volumes=volumes,
        selected_volume=selected,
        validated_root=validated_root,
        config_project_root=config_project_root,
        status=status,
        blocking_reasons=blocking_reasons,
        created_or_verified_dirs=created_or_verified_dirs,
        generated_files=generated_files,
        python_info=python_info,
        pip_info=pip_info,
        gsutil_info=gsutil_info,
    )

    local_report.parent.mkdir(parents=True, exist_ok=True)
    local_report.write_text(report, encoding="utf-8")

    if external_report is not None:
        external_report.parent.mkdir(parents=True, exist_ok=True)
        external_report.write_text(report, encoding="utf-8")

    selected_payload = asdict(selected) if selected is not None else None
    status_payload: dict[str, object] = {
        "current_step": "0R",
        "status": status,
        "legacy_external_disk_label": LEGACY_LABEL,
        "label_requirement_disabled": True,
        "validated_external_root": format_path(validated_root),
        "external_disk_found": selected is not None,
        "external_disk": selected_payload,
        "configured_project_root": format_path(config_project_root) if config_project_root else None,
        "filesystem_acceptable": (
            selected.filesystem.upper() in ACCEPTABLE_FILESYSTEMS if selected else False
        ),
        "old_partition_limit_cleared": (
            selected.total_bytes > OLD_PARTITION_LIMIT_BYTES if selected else False
        ),
        "external_capacity_confirmed": (
            selected.total_bytes >= EXPECTED_EXTERNAL_TOTAL_BYTES if selected else False
        ),
        "sen1floods11_estimated_size_gb": bytes_to_gb(SEN1FLOODS11_ESTIMATED_BYTES),
        "safety_reserve_gb": bytes_to_gb(SAFETY_RESERVE_BYTES),
        "next_step_allowed": False,
        "human_validation_required": True,
        "generated_at": generated_at,
        "blocking_reasons": blocking_reasons,
    }
    write_json(local_status, status_payload)
    if external_status is not None:
        write_json(external_status, status_payload)

    print("\nSTEP 0R - Environment refresh summary")
    print(f"Validated external root: {format_path(validated_root)}")
    print("Label requirement: disabled by explicit human validation")
    print(f"Status: {status}")
    for volume in volumes:
        print(
            f"- {volume.root} label={volume.label or '<empty>'} "
            f"fs={volume.filesystem} total={bytes_to_gb(volume.total_bytes)} GB "
            f"free={bytes_to_gb(volume.free_bytes)} GB"
        )
    if blocking_reasons:
        print("Blocking reasons:")
        for reason in blocking_reasons:
            print(f"- {reason}")
        print("STOP: STEP 1 was not started.")
    else:
        print("Storage is valid for STEP 1.")

    return 0 if status == "done" else 2


if __name__ == "__main__":
    raise SystemExit(main())
