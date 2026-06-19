import csv
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.create_balanced_manifest_split import main


def test_balanced_manifest_split(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    stats_path = tmp_path / "stats.csv"
    output_path = tmp_path / "output.csv"

    # Create dummy manifest (8 samples)
    fieldnames = ["sample_id", "split", "image_path", "mask_path", "dem_path"]
    manifest_rows = [
        {
            "sample_id": f"s{i}",
            "split": "train",
            "image_path": "a",
            "mask_path": "b",
            "dem_path": "c",
        }
        for i in range(8)
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Create dummy stats (4 with water, 4 without)
    stats_fieldnames = [
        "sample_id",
        "positive_pixel_count",
        "valid_pixel_count",
        "target_positive_rate",
    ]
    stats_rows = []
    for i in range(4):
        stats_rows.append(
            {
                "sample_id": f"s{i}",
                "positive_pixel_count": "100",
                "valid_pixel_count": "1000",
                "target_positive_rate": "0.1",
            }
        )
    for i in range(4, 8):
        stats_rows.append(
            {
                "sample_id": f"s{i}",
                "positive_pixel_count": "0",
                "valid_pixel_count": "1000",
                "target_positive_rate": "0.0",
            }
        )

    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    test_args = [
        "script",
        "--manifest",
        str(manifest_path),
        "--stats-csv",
        str(stats_path),
        "--output-manifest",
        str(output_path),
        "--val-ratio",
        "0.25",
        "--min-val-samples",
        "2",
        "--seed",
        "42",
    ]

    with patch.object(sys, "argv", test_args):
        ret = main()
        assert ret == 0

    # Verify output
    assert output_path.exists()
    with output_path.open("r", newline="", encoding="utf-8") as f:
        output_rows = list(csv.DictReader(f))

    assert len(output_rows) == 8
    assert list(output_rows[0].keys()) == fieldnames, "Columns must be conserved."

    val_count = sum(1 for r in output_rows if r["split"] == "val")
    train_count = sum(1 for r in output_rows if r["split"] == "train")

    # 0.25 of 8 is 2. min is 2. So we expect 2 val and 6 train.
    assert val_count == 2
    assert train_count == 6
