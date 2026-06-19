"""STEP 1A: check Sen1Floods11 source and storage feasibility.

This script inspects remote metadata only. It does not download dataset files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
import yaml


BUCKET = "sen1floods11"
GCS_URI = f"gs://{BUCKET}"
GCS_HTTPS_URL = f"https://storage.googleapis.com/{BUCKET}/"
OFFICIAL_REPO_URL = "https://github.com/cloudtostreet/Sen1Floods11"
OFFICIAL_PAPER_URL = (
    "https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/"
    "Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_"
    "CVPRW_2020_paper.html"
)
FAT32_SINGLE_FILE_LIMIT_BYTES = 4 * 1024**3
VALIDATION_TOTAL_THRESHOLD_BYTES = 25 * 1_000_000_000
MIN_FREE_AFTER_DOWNLOAD_BYTES = 5 * 1_000_000_000


@dataclass(frozen=True)
class RemoteObject:
    uri: str
    size_bytes: int


@dataclass(frozen=True)
class FeasibilityResult:
    status: str
    reasons: list[str]
    total_bytes: int
    object_count: int
    max_object: RemoteObject | None
    max_geotiff: RemoteObject | None
    files_over_fat32_limit: list[RemoteObject]
    free_space_bytes: int
    filesystem_label: str | None
    filesystem_type: str | None
    category_counts: Counter[str]
    category_sizes: Counter[str]
    extension_counts: Counter[str]
    extension_sizes: Counter[str]


def _format_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def bytes_to_gb(value: int | float) -> float:
    return round(float(value) / 1_000_000_000, 3)


def bytes_to_gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")
    return payload


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


def list_objects_with_gsutil(gsutil: str, timeout: float) -> list[RemoteObject]:
    command = [gsutil, "ls", "-l", "-r", f"{GCS_URI}/**"]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not list Sen1Floods11 object metadata with gsutil.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr: {completed.stderr.strip()}"
        )

    objects: list[RemoteObject] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[2].startswith("gs://"):
            objects.append(RemoteObject(uri=parts[2], size_bytes=int(parts[0])))
    return objects


def list_objects_with_json_api(timeout: float) -> list[RemoteObject]:
    objects: list[RemoteObject] = []
    page_token = None
    session = requests.Session()
    while True:
        params = {
            "fields": "items(name,size),nextPageToken",
            "maxResults": "1000",
        }
        if page_token:
            params["pageToken"] = page_token
        response = session.get(
            f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("items", []):
            name = item["name"]
            size = int(item.get("size") or 0)
            objects.append(RemoteObject(uri=f"{GCS_URI}/{name}", size_bytes=size))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return objects


def category_for_uri(uri: str) -> str:
    relative = uri.removeprefix(f"{GCS_URI}/")
    parts = relative.split("/")
    if relative.startswith("v1.1/data/flood_events/") and len(parts) >= 4:
        return "/".join(parts[:4])
    if relative.startswith("v1.1/data/perm_water/") and len(parts) >= 4:
        return "/".join(parts[:4])
    if relative.startswith("v1.1/catalog/"):
        return "v1.1/catalog"
    if relative.startswith("v1.1/splits/") and len(parts) >= 3:
        return "/".join(parts[:3])
    if relative.startswith("v1.1/checkpoints/"):
        return "v1.1/checkpoints"
    return relative


def extension_for_uri(uri: str) -> str:
    name = uri.rsplit("/", maxsplit=1)[-1]
    if "." not in name:
        return "<none>"
    return "." + name.rsplit(".", maxsplit=1)[-1].lower()


def inspect_feasibility(
    *,
    objects: list[RemoteObject],
    free_space_bytes: int,
    filesystem_label: str | None,
    filesystem_type: str | None,
) -> FeasibilityResult:
    if not objects:
        return FeasibilityResult(
            status="blocked",
            reasons=["No remote objects were discovered in the source bucket."],
            total_bytes=0,
            object_count=0,
            max_object=None,
            max_geotiff=None,
            files_over_fat32_limit=[],
            free_space_bytes=free_space_bytes,
            filesystem_label=filesystem_label,
            filesystem_type=filesystem_type,
            category_counts=Counter(),
            category_sizes=Counter(),
            extension_counts=Counter(),
            extension_sizes=Counter(),
        )

    total_bytes = sum(item.size_bytes for item in objects)
    max_object = max(objects, key=lambda item: item.size_bytes)
    geotiffs = [item for item in objects if item.uri.lower().endswith((".tif", ".tiff"))]
    max_geotiff = max(geotiffs, key=lambda item: item.size_bytes) if geotiffs else None
    files_over_fat32_limit = [
        item for item in objects if item.size_bytes > FAT32_SINGLE_FILE_LIMIT_BYTES
    ]

    category_counts: Counter[str] = Counter()
    category_sizes: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    extension_sizes: Counter[str] = Counter()
    for item in objects:
        category = category_for_uri(item.uri)
        extension = extension_for_uri(item.uri)
        category_counts[category] += 1
        category_sizes[category] += item.size_bytes
        extension_counts[extension] += 1
        extension_sizes[extension] += item.size_bytes

    reasons: list[str] = []
    if files_over_fat32_limit:
        reasons.append("At least one remote object exceeds the FAT32 single-file limit.")
    if total_bytes > VALIDATION_TOTAL_THRESHOLD_BYTES:
        reasons.append("Estimated total dataset size exceeds the 25 GB validation threshold.")
    if free_space_bytes - total_bytes < MIN_FREE_AFTER_DOWNLOAD_BYTES:
        reasons.append("Free space after a complete download would be below 5 GB.")
    if filesystem_type and filesystem_type.upper() == "FAT32":
        reasons.append("External disk is FAT32; individual files are compatible, but exFAT/NTFS is safer.")

    blocking = any(
        reason
        for reason in reasons
        if reason.startswith("At least one")
        or reason.startswith("Estimated total")
        or reason.startswith("Free space")
    )
    status = "blocked" if blocking else "done"

    return FeasibilityResult(
        status=status,
        reasons=reasons,
        total_bytes=total_bytes,
        object_count=len(objects),
        max_object=max_object,
        max_geotiff=max_geotiff,
        files_over_fat32_limit=files_over_fat32_limit,
        free_space_bytes=free_space_bytes,
        filesystem_label=filesystem_label,
        filesystem_type=filesystem_type,
        category_counts=category_counts,
        category_sizes=category_sizes,
        extension_counts=extension_counts,
        extension_sizes=extension_sizes,
    )


def build_markdown_report(
    *,
    generated_at: str,
    repo_root: Path,
    config_path: Path,
    external_project_root: Path,
    raw_target: Path,
    reports_dir: Path,
    logs_dir: Path,
    source_method: str,
    result: FeasibilityResult,
    local_report: Path,
    external_report: Path,
    status_path: Path,
) -> str:
    max_object = result.max_object
    max_geotiff = result.max_geotiff
    free_after = result.free_space_bytes - result.total_bytes
    fat32_compatible = len(result.files_over_fat32_limit) == 0
    disk_sufficient = free_after >= MIN_FREE_AFTER_DOWNLOAD_BYTES
    total_needs_validation = result.total_bytes > VALIDATION_TOTAL_THRESHOLD_BYTES

    lines = [
        "# STEP 1A - Sen1Floods11 download feasibility report",
        "",
        "## Summary",
        f"- Status: `{result.status}`",
        f"- Generated at: `{generated_at}`",
        "- Download performed: `false`",
        "- Next step allowed: `false`",
        "",
        "## Official source checked",
        f"- Official repository: {OFFICIAL_REPO_URL}",
        f"- Paper page: {OFFICIAL_PAPER_URL}",
        f"- Public Google Cloud Storage bucket: `{GCS_URI}`",
        f"- HTTPS bucket endpoint: {GCS_HTTPS_URL}",
        f"- Metadata inspection method: `{source_method}`",
        "- Official recommended full-download command: "
        f"`gsutil -m rsync -r {GCS_URI} <local_directory>`",
        "",
        "## Storage target",
        f"- Local repo root: `{_format_path(repo_root)}`",
        f"- Config file: `{_format_path(config_path)}`",
        f"- External project root: `{_format_path(external_project_root)}`",
        f"- Raw target directory: `{_format_path(raw_target)}`",
        f"- Reports directory: `{_format_path(reports_dir)}`",
        f"- Logs directory: `{_format_path(logs_dir)}`",
        f"- Filesystem label: `{result.filesystem_label or 'unknown'}`",
        f"- Filesystem type: `{result.filesystem_type or 'unknown'}`",
        f"- Current free space: `{bytes_to_gb(result.free_space_bytes)} GB` "
        f"(`{bytes_to_gib(result.free_space_bytes)} GiB`)",
        "",
        "## Remote size analysis",
        f"- Remote object count: `{result.object_count}`",
        f"- Estimated total size: `{bytes_to_gb(result.total_bytes)} GB` "
        f"(`{bytes_to_gib(result.total_bytes)} GiB`)",
        f"- Total size exceeds 25 GB threshold: `{str(total_needs_validation).lower()}`",
    ]

    if max_object:
        lines.extend(
            [
                f"- Largest individual object: `{bytes_to_gb(max_object.size_bytes)} GB` "
                f"(`{bytes_to_gib(max_object.size_bytes)} GiB`)",
                f"- Largest individual object URI: `{max_object.uri}`",
            ]
        )
    if max_geotiff:
        lines.extend(
            [
                f"- Largest GeoTIFF object: `{bytes_to_gb(max_geotiff.size_bytes)} GB` "
                f"(`{bytes_to_gib(max_geotiff.size_bytes)} GiB`)",
                f"- Largest GeoTIFF URI: `{max_geotiff.uri}`",
            ]
        )

    lines.extend(
        [
            f"- Objects larger than FAT32 4 GiB limit: `{len(result.files_over_fat32_limit)}`",
            f"- FAT32 compatibility by individual file size: `{str(fat32_compatible).lower()}`",
            f"- Estimated free space after complete download: `{bytes_to_gb(free_after)} GB` "
            f"(`{bytes_to_gib(free_after)} GiB`)",
            f"- Disk space sufficient for full download and 5 GB reserve: `{str(disk_sufficient).lower()}`",
            "",
            "## Size by main source category",
            "| Category | Objects | Size GB |",
            "|---|---:|---:|",
        ]
    )
    for category, size in sorted(
        result.category_sizes.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"| `{category}` | {result.category_counts[category]} | {bytes_to_gb(size)} |")

    lines.extend(["", "## Size by extension", "| Extension | Objects | Size GB |", "|---|---:|---:|"])
    for extension, size in sorted(
        result.extension_sizes.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"| `{extension}` | {result.extension_counts[extension]} | {bytes_to_gb(size)} |")

    lines.extend(
        [
            "",
            "## Format and download options",
            "- The bucket exposes many individual files, mainly Cloud-Optimized GeoTIFF-like `.tif` objects, plus STAC catalog JSON files, split CSV files, and small checkpoints.",
            "- No single full-dataset archive was identified. `catalog.zip` is a small metadata archive, not the full imagery dataset.",
            "- A file-by-file or collection-by-collection download is preferable for FAT32 because the largest object is far below 4 GiB.",
            "- For a constrained disk, the most realistic first option is to download metadata, splits, and the hand-labeled subset before considering the weakly labeled subset.",
            "",
            "## Problems detected",
        ]
    )
    if result.reasons:
        lines.extend(f"- {reason}" for reason in result.reasons)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Recommendation",
        ]
    )
    if not fat32_compatible:
        lines.append("- Do not download to this FAT32 disk because at least one file exceeds 4 GiB.")
    elif not disk_sufficient or total_needs_validation:
        lines.extend(
            [
                "- Do not launch the full Sen1Floods11 download now.",
                "- The full bucket is larger than the configured validation threshold and larger than the currently available free space with a 5 GB reserve.",
                "- Recommended options: use a larger disk formatted as exFAT or NTFS, free substantial space, or validate a smaller scoped download first.",
                "- FAT32 is not blocked by individual file size for this source, but exFAT/NTFS is preferable for reliability and future large artifacts.",
            ]
        )
    else:
        lines.append("- Storage appears feasible, but Step 1B still requires human validation.")

    lines.extend(
        [
            "",
            "## Generated files",
            f"- Local report: `{_format_path(local_report)}`",
            f"- External report: `{_format_path(external_report)}`",
            f"- Pipeline status: `{_format_path(status_path)}`",
            "",
            "## Proposed next step",
            "Human validation is required before STEP 1B. STEP 1B should either prepare a smaller scoped download or move to a larger exFAT/NTFS storage target.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def filesystem_info(path: Path) -> tuple[str | None, str | None]:
    if sys.platform != "win32":
        return None, None

    try:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$drive = (Get-Item -LiteralPath "
                f"'{path.anchor}'"
                ").PSDrive.Name; "
                "Get-Volume -DriveLetter $drive | "
                "Select-Object -ExpandProperty FileSystemLabel; "
                "Get-Volume -DriveLetter $drive | "
                "Select-Object -ExpandProperty FileSystem"
            ),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
        values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        label = values[0] if values else None
        fstype = values[1] if len(values) > 1 else None
        return label, fstype
    except Exception:
        return None, None


def print_terminal_summary(result: FeasibilityResult) -> None:
    max_object = result.max_object
    free_after = result.free_space_bytes - result.total_bytes
    fat32_compatible = len(result.files_over_fat32_limit) == 0
    disk_sufficient = free_after >= MIN_FREE_AFTER_DOWNLOAD_BYTES

    if not fat32_compatible:
        recommendation = "changer de disque ou reformater en exFAT/NTFS"
    elif not disk_sufficient:
        recommendation = "ne pas telecharger maintenant; changer de disque ou liberer de l'espace"
    elif result.total_bytes > VALIDATION_TOTAL_THRESHOLD_BYTES:
        recommendation = "demander validation avant telechargement"
    else:
        recommendation = "telechargement techniquement possible apres validation humaine"

    print("\nSTEP 1A - Sen1Floods11 feasibility summary")
    print(f"Source officielle trouvee: {OFFICIAL_REPO_URL}")
    print(f"URL/methode de telechargement: gsutil -m rsync -r {GCS_URI} <local_directory>")
    print(f"Taille totale estimee: {bytes_to_gb(result.total_bytes)} GB")
    if max_object:
        print(
            "Taille maximale d'un fichier individuel: "
            f"{bytes_to_gb(max_object.size_bytes)} GB ({max_object.uri})"
        )
    else:
        print("Taille maximale d'un fichier individuel: inconnue")
    print(f"Compatibilite FAT32: {'oui' if fat32_compatible else 'non'}")
    print(f"Espace disque suffisant: {'oui' if disk_sufficient else 'non'}")
    print(f"Espace libre apres telechargement complet estime: {bytes_to_gb(free_after)} GB")
    print(f"Recommendation: {recommendation}")
    print("STOP: validation humaine requise avant STEP 1B.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sen1floods11.yaml"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--metadata-backend",
        choices=["auto", "gsutil", "json-api"],
        default="auto",
        help="Remote metadata listing backend. No dataset files are downloaded.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)

    external_project_root = Path(str(config["project_root"]))
    raw_target = Path(str(config["raw"]["sen1floods11"]))  # type: ignore[index]
    reports_dir = Path(str(config["reports"]))
    logs_dir = Path(str(config["logs"]))
    external_status = external_project_root / "pipeline_status.json"
    local_status = repo_root / "pipeline_status.json"
    local_report = repo_root / "reports" / "STEP_1A_download_feasibility_report.md"
    external_report = reports_dir / "STEP_1A_download_feasibility_report.md"
    log_path = logs_dir / "01A_sen1floods11_feasibility.log"

    generated_at = datetime.now().isoformat(timespec="seconds")
    gsutil = find_gsutil()
    source_method = ""

    if args.metadata_backend in {"auto", "gsutil"} and gsutil is not None:
        objects = list_objects_with_gsutil(gsutil, args.timeout)
        source_method = f"gsutil metadata listing via {gsutil}"
    elif args.metadata_backend == "gsutil":
        raise SystemExit("gsutil backend requested, but gsutil was not found.")
    else:
        objects = list_objects_with_json_api(args.timeout)
        source_method = "Google Cloud Storage JSON API object metadata listing"

    disk_usage = shutil.disk_usage(external_project_root.anchor or external_project_root)
    filesystem_label, filesystem_type = filesystem_info(external_project_root)
    result = inspect_feasibility(
        objects=objects,
        free_space_bytes=disk_usage.free,
        filesystem_label=filesystem_label,
        filesystem_type=filesystem_type,
    )

    status_payload: dict[str, object] = {
        "current_step": "1A",
        "status": result.status,
        "external_disk_label": config.get("external_disk_label"),
        "external_disk_path": config.get("external_disk_path"),
        "external_project_root": _format_path(external_project_root),
        "source_official_repository": OFFICIAL_REPO_URL,
        "source_gcs_uri": GCS_URI,
        "metadata_method": source_method,
        "dataset_total_bytes": result.total_bytes,
        "dataset_total_gb": bytes_to_gb(result.total_bytes),
        "object_count": result.object_count,
        "max_file_bytes": result.max_object.size_bytes if result.max_object else None,
        "max_file_gb": bytes_to_gb(result.max_object.size_bytes) if result.max_object else None,
        "max_file_uri": result.max_object.uri if result.max_object else None,
        "files_over_fat32_limit": len(result.files_over_fat32_limit),
        "fat32_compatible_by_file_size": len(result.files_over_fat32_limit) == 0,
        "free_space_gb": bytes_to_gb(result.free_space_bytes),
        "free_space_after_full_download_gb": bytes_to_gb(
            result.free_space_bytes - result.total_bytes
        ),
        "disk_space_sufficient_for_full_download": (
            result.free_space_bytes - result.total_bytes >= MIN_FREE_AFTER_DOWNLOAD_BYTES
        ),
        "next_step_allowed": False,
        "human_validation_required": True,
        "generated_at": generated_at,
        "blocking_reasons": result.reasons,
    }

    report = build_markdown_report(
        generated_at=generated_at,
        repo_root=repo_root,
        config_path=config_path,
        external_project_root=external_project_root,
        raw_target=raw_target,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
        source_method=source_method,
        result=result,
        local_report=local_report,
        external_report=external_report,
        status_path=local_status,
    )

    local_report.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    local_report.write_text(report, encoding="utf-8")
    external_report.write_text(report, encoding="utf-8")
    write_json(local_status, status_payload)
    write_json(external_status, status_payload)
    log_path.write_text(
        (
            f"{generated_at} STEP 1A status={result.status} "
            f"objects={result.object_count} total_bytes={result.total_bytes} "
            f"max_file_bytes={result.max_object.size_bytes if result.max_object else 'NA'}\n"
        ),
        encoding="utf-8",
    )

    print_terminal_summary(result)
    return 0 if result.status in {"done", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
