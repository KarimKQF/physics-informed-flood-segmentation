from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced manifest split.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stats-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--min-val-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def calculate_rate(samples: list[dict[str, object]]) -> float:
    pos = sum(float(s["positive_pixel_count"]) for s in samples)
    valid = sum(float(s["valid_pixel_count"]) for s in samples)
    return pos / valid if valid > 0 else 0.0


def main() -> int:
    args = parse_args()
    try:
        with args.manifest.open("r", newline="", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
            if not manifest_rows:
                print("[ERROR] Empty manifest", file=sys.stderr)
                return 1
            fieldnames = list(manifest_rows[0].keys())

        with args.stats_csv.open("r", newline="", encoding="utf-8") as f:
            stats_rows = {row["sample_id"]: row for row in csv.DictReader(f)}

        # Attach stats to manifest rows
        for row in manifest_rows:
            sample_id = row["sample_id"]
            if sample_id not in stats_rows:
                raise ValueError(f"Sample {sample_id} not found in stats CSV.")
            row["positive_pixel_count"] = float(stats_rows[sample_id]["positive_pixel_count"])
            row["valid_pixel_count"] = float(stats_rows[sample_id]["valid_pixel_count"])
            row["target_positive_rate"] = float(stats_rows[sample_id]["target_positive_rate"])

        total_samples = len(manifest_rows)
        num_val = max(args.min_val_samples, int(round(total_samples * args.val_ratio)))

        if total_samples <= num_val:
            print("[ERROR] Not enough samples to create validation split.", file=sys.stderr)
            return 1

        random.seed(args.seed)

        best_diff = float("inf")
        best_split = None

        # Random search for the best split
        indices = list(range(total_samples))

        # Limit trials to avoid hanging on large datasets
        num_trials = 10000 if total_samples < 50 else 100
        for _ in range(num_trials):
            random.shuffle(indices)
            val_indices = set(indices[:num_val])

            val_samples = [manifest_rows[i] for i in range(total_samples) if i in val_indices]
            train_samples = [manifest_rows[i] for i in range(total_samples) if i not in val_indices]

            train_rate = calculate_rate(train_samples)
            val_rate = calculate_rate(val_samples)
            diff = abs(train_rate - val_rate)

            if diff < best_diff:
                best_diff = diff
                best_split = val_indices

        # Apply the best split
        train_final = []
        val_final = []
        for i, row in enumerate(manifest_rows):
            # Clean up added keys
            del row["positive_pixel_count"]
            del row["valid_pixel_count"]
            del row["target_positive_rate"]

            if i in best_split:
                row["split"] = "val"
                val_final.append(row)
            else:
                row["split"] = "train"
                train_final.append(row)

        # Calculate stats on final split for reporting
        val_samples_stats = [stats_rows[r["sample_id"]] for r in val_final]
        train_samples_stats = [stats_rows[r["sample_id"]] for r in train_final]

        train_rate_final = calculate_rate(train_samples_stats)
        val_rate_final = calculate_rate(val_samples_stats)

        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.output_manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        print(f"[INFO] Train samples: {len(train_final)}")
        print(f"[INFO] Val samples: {len(val_final)}")
        print(f"[INFO] target_positive_rate_train: {train_rate_final:.6f}")
        print(f"[INFO] target_positive_rate_val: {val_rate_final:.6f}")
        print(f"[INFO] difference: {abs(train_rate_final - val_rate_final):.6f}")

        if total_samples <= 8:
            print(
                "[WARNING] The subset of 8 samples is too small to obtain a truly balanced split."
            )

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
