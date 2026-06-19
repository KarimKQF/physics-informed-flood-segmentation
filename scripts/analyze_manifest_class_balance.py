from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze class balance per sample in a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def analyze_sample(row: dict[str, str]) -> dict[str, object]:
    sample_id = row.get("sample_id", "unknown")
    split = row.get("split", "unknown")
    image_path = row.get("image_path", "")
    mask_path = row.get("mask_path", "")
    dem_path = row.get("dem_path", "")

    mask_file = Path(mask_path)
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask not found: {mask_file}")

    with rasterio.open(mask_file) as src:
        mask = src.read(1)

    pos = int((mask == 1).sum())
    neg = int((mask == 0).sum())
    invalid = int((mask == -1).sum())
    valid = pos + neg

    total_pixels = mask.size
    valid_ratio = valid / total_pixels if total_pixels > 0 else 0.0
    target_positive_rate = pos / valid if valid > 0 else 0.0

    return {
        "sample_id": sample_id,
        "split": split,
        "valid_pixel_count": valid,
        "positive_pixel_count": pos,
        "negative_pixel_count": neg,
        "invalid_pixel_count": invalid,
        "valid_ratio": valid_ratio,
        "target_positive_rate": target_positive_rate,
        "image_path": image_path,
        "mask_path": mask_path,
        "dem_path": dem_path,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "valid_pixel_count",
        "positive_pixel_count",
        "negative_pixel_count",
        "invalid_pixel_count",
        "valid_ratio",
        "target_positive_rate",
        "image_path",
        "mask_path",
        "dem_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    global_pos = sum(r["positive_pixel_count"] for r in rows)
    global_valid = sum(r["valid_pixel_count"] for r in rows)
    global_rate = global_pos / global_valid if global_valid > 0 else 0.0

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    train_pos = sum(r["positive_pixel_count"] for r in train_rows)
    train_valid = sum(r["valid_pixel_count"] for r in train_rows)
    train_rate = train_pos / train_valid if train_valid > 0 else 0.0

    val_pos = sum(r["positive_pixel_count"] for r in val_rows)
    val_valid = sum(r["valid_pixel_count"] for r in val_rows)
    val_rate = val_pos / val_valid if val_valid > 0 else 0.0

    samples_with_water = sum(1 for r in rows if r["target_positive_rate"] > 0)
    samples_no_water = sum(1 for r in rows if r["target_positive_rate"] == 0)

    lines = [
        "# Class Balance Report",
        "",
        "## Summary",
        f"- **Global target_positive_rate**: {global_rate:.6f}",
        f"- **Train target_positive_rate**: {train_rate:.6f}",
        f"- **Val target_positive_rate**: {val_rate:.6f}",
        f"- **Samples with water**: {samples_with_water}",
        f"- **Samples quasi sans eau**: {samples_no_water}",
        "",
        "## Per-Sample Statistics",
        "",
        "| Sample ID | Split | Valid Pixels | Positive | target_positive_rate |",
        "|---|---|---:|---:|---:|",
    ]

    # Sort by positive rate descending
    sorted_rows = sorted(rows, key=lambda x: x["target_positive_rate"], reverse=True)
    for r in sorted_rows:
        lines.append(
            f"| {r['sample_id']} | {r['split']} | {r['valid_pixel_count']} | "
            f"{r['positive_pixel_count']} | {r['target_positive_rate']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        with args.manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)

        results = [analyze_sample(row) for row in manifest_rows]

        write_csv(args.output_csv, results)
        write_md(args.output_md, results)

        print(f"[OK] Wrote CSV: {args.output_csv}")
        print(f"[OK] Wrote MD: {args.output_md}")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
