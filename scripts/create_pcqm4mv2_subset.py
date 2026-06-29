from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from mkd_mpp.pcqm4mv2 import load_official_split


def choose_indices_by_official_split(
    rows: list[dict[str, str]],
    split_sets: dict[str, set[str]],
    sample_size: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    split_order = ["train", "valid", "test-dev", "test-challenge"]
    ratios = {"train": 0.90, "valid": 0.02, "test-dev": 0.04, "test-challenge": 0.04}
    rows_by_split = {name: [] for name in split_order}

    for row in rows:
        idx = row["idx"]
        for split_name in split_order:
            if idx in split_sets[split_name]:
                rows_by_split[split_name].append(row)
                break

    selected = []
    remaining = sample_size
    for split_name in split_order:
        if split_name == split_order[-1]:
            target = remaining
        else:
            target = round(sample_size * ratios[split_name])
            remaining -= target
        pool = rows_by_split[split_name]
        if target > len(pool):
            raise ValueError(f"Not enough rows for split {split_name}: need {target}, have {len(pool)}")
        sampled = rng.sample(pool, target)
        for row in sampled:
            row = dict(row)
            row["split"] = split_name
            selected.append(row)

    selected.sort(key=lambda row: int(row["idx"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PCQM4Mv2 subset preserving official split ratios.")
    parser.add_argument("--input-csv", default="dataset/data.csv")
    parser.add_argument("--output-csv", default="dataset/data_100k.csv")
    parser.add_argument("--split-root", default="dataset")
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if "idx" not in fieldnames:
        raise ValueError("Input CSV must contain idx for official split matching.")
    if "split" not in fieldnames:
        fieldnames.append("split")

    selected = choose_indices_by_official_split(
        rows=rows,
        split_sets=load_official_split(args.split_root),
        sample_size=args.sample_size,
        seed=args.seed,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)


if __name__ == "__main__":
    main()
