from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import rasterio


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"sample_id", "image_path", "mask_path", "dem_path", "split"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        return list(reader)


def inspect_bounds(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")

    samples: list[dict[str, Any]] = []
    crs_values: set[str] = set()
    shapes: set[tuple[int, int]] = set()
    global_left = float("inf")
    global_bottom = float("inf")
    global_right = float("-inf")
    global_top = float("-inf")

    for row in rows:
        sample_id = row["sample_id"]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist for {sample_id}: {image_path}")

        with rasterio.open(image_path) as image:
            bounds = image.bounds
            crs = str(image.crs)
            transform_values = list(tuple(image.transform)[:6])
            resolution = [float(image.res[0]), float(image.res[1])]
            width = image.width
            height = image.height

        crs_values.add(crs)
        shapes.add((height, width))
        global_left = min(global_left, bounds.left)
        global_bottom = min(global_bottom, bounds.bottom)
        global_right = max(global_right, bounds.right)
        global_top = max(global_top, bounds.top)

        samples.append(
            {
                "sample_id": sample_id,
                "image_path": image_path.as_posix(),
                "crs": crs,
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "width": width,
                "height": height,
                "transform": transform_values,
                "resolution": resolution,
            }
        )

    payload: dict[str, Any] = {
        "num_samples": len(samples),
        "crs_list": sorted(crs_values),
        "shapes": [{"height": height, "width": width} for height, width in sorted(shapes)],
        "global_bounds": {
            "left": global_left,
            "bottom": global_bottom,
            "right": global_right,
            "top": global_top,
        },
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect image bounds in a Sen1Floods11 manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        payload = inspect_bounds(args.manifest, args.output)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Bounds report written to: {args.output}")
    print(f"[INFO] Samples: {payload['num_samples']}")
    print(f"[INFO] CRS list: {payload['crs_list']}")
    print(f"[INFO] Shapes: {payload['shapes']}")
    print(f"[INFO] Global bounds: {payload['global_bounds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
