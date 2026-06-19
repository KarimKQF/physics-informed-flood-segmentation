from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.utils import ensure_external_data_tree, load_local_paths  # noqa: E402

RASTER_SUFFIXES = (".tif", ".tiff")
MASK_KEYWORDS = ("label", "mask", "water", "labelhand", "s1otsulabel", "jrcwater")
BUCKETS_TO_TEST = (
    "gs://senfloods11/",
    "gs://sen1floods11/",
    "gs://sen1floods11/v1.1/",
)
RECURSIVE_TARGETS = (
    "gs://sen1floods11/v1.1/data/**",
    "gs://sen1floods11/v1.1/catalog/**",
)


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


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


def run_command(command: list[str], timeout: float) -> CommandResult:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def write_result(file, result: CommandResult, max_lines: int) -> list[str]:
    file.write(f"$ {' '.join(result.command)}\n")
    file.write(f"returncode: {result.returncode}\n")
    if result.stderr.strip():
        file.write("[stderr]\n")
        file.write(result.stderr)
        if not result.stderr.endswith("\n"):
            file.write("\n")
    lines = result.stdout.splitlines()
    if lines:
        file.write("[stdout]\n")
        for line in lines[:max_lines]:
            file.write(f"{line}\n")
        if len(lines) > max_lines:
            file.write(f"... truncated after {max_lines} lines out of {len(lines)}\n")
    file.write("\n")
    return lines


def summarize_listing(lines: list[str]) -> dict[str, object]:
    geotiffs = [line for line in lines if line.lower().endswith(RASTER_SUFFIXES)]
    masks = [
        line
        for line in geotiffs
        if any(keyword in line.rsplit("/", maxsplit=1)[-1].lower() for keyword in MASK_KEYWORDS)
    ]
    buckets = sorted({line.split("/", 3)[2] for line in lines if line.startswith("gs://")})
    return {
        "line_count": len(lines),
        "geotiff_count": len(geotiffs),
        "mask_like_geotiff_count": len(masks),
        "buckets": buckets,
        "example_geotiffs": geotiffs[:20],
        "example_masks": masks[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect remote Sen1Floods11 GCS paths.")
    parser.add_argument("--config", type=Path, default=Path("configs/local_paths.yaml"))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-lines", type=int, default=5000)
    args = parser.parse_args()

    try:
        paths = load_local_paths(args.config)
        ensure_external_data_tree(paths)
        log_path = paths.logs / "sen1floods11_remote_listing.txt"
        gsutil = find_gsutil()

        with log_path.open("w", encoding="utf-8") as log:
            log.write("Sen1Floods11 remote inspection\n\n")
            if gsutil is None:
                message = (
                    "gsutil was not found. Install Google Cloud SDK, close/reopen the "
                    "terminal, then test: gsutil version"
                )
                log.write(f"[ERROR] {message}\n")
                print(f"[ERROR] {message}")
                print(f"[INFO] Report written to: {log_path}")
                return 1

            log.write(f"[INFO] gsutil: {gsutil}\n\n")
            all_lines: list[str] = []
            for uri in BUCKETS_TO_TEST:
                result = run_command([gsutil, "ls", uri], timeout=args.timeout)
                lines = write_result(log, result, args.max_lines)
                all_lines.extend(lines)

            for uri in RECURSIVE_TARGETS:
                try:
                    result = run_command([gsutil, "ls", "-r", uri], timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    log.write(f"$ {gsutil} ls -r {uri}\n")
                    log.write(f"[ERROR] command timed out after {args.timeout} seconds\n\n")
                    continue
                lines = write_result(log, result, args.max_lines)
                all_lines.extend(lines)

            summary = summarize_listing(all_lines)
            log.write("[summary]\n")
            for key, value in summary.items():
                log.write(f"{key}: {value}\n")

        print(f"[OK] Remote listing report written to: {log_path}")
        print(f"[INFO] GeoTIFF lines detected: {summary['geotiff_count']}")
        print(f"[INFO] Mask-like GeoTIFF lines detected: {summary['mask_like_geotiff_count']}")
        print("[INFO] Best candidate bucket/path: gs://sen1floods11/v1.1/")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
