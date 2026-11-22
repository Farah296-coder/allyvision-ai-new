"""Create deterministic train/val/test splits without separating video sequences."""
import argparse
import csv
import hashlib
from pathlib import Path


def split_for_group(group: str, seed: int, train: float, val: float) -> str:
    if not group.strip():
        raise ValueError("Each record must have a non-empty sequence_id")
    point = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
    return "train" if point < train else ("val" if point < train + val else "test")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_csv", type=Path, help="CSV with image_path and sequence_id columns")
    p.add_argument("output_csv", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=.70)
    p.add_argument("--val", type=float, default=.15)
    args = p.parse_args()
    if args.train <= 0 or args.val <= 0 or args.train + args.val >= 1:
        p.error("train and val must be positive and sum to less than 1")
    with args.input_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"image_path", "sequence_id"}.issubset(rows[0]):
        p.error("input CSV must have rows and image_path, sequence_id columns")
    group_splits = {r["sequence_id"]: split_for_group(r["sequence_id"], args.seed, args.train, args.val) for r in rows}
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[*rows[0].keys(), "split"])
        w.writeheader()
        for row in rows:
            row["split"] = group_splits[row["sequence_id"]]
            w.writerow(row)
    counts = {s: sum(v == s for v in group_splits.values()) for s in ("train", "val", "test")}
    print(f"Wrote {len(rows)} frames from {len(group_splits)} sequences to {args.output_csv}")
    print("Sequence counts:", counts)


if __name__ == "__main__":
    main()
