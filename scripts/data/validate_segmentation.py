"""Validate paired indexed masks and disjoint sequence-level split manifests."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path, help="CSV with image_path,mask_path,sequence_id,split")
    p.add_argument("--num-classes", type=int, default=18)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--check-paths", action="store_true")
    a = p.parse_args()
    with a.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"image_path", "mask_path", "sequence_id", "split"}
    if not rows or not required.issubset(rows[0]):
        p.error(f"manifest needs columns {sorted(required)}")
    errors, group_split = [], {}
    for i, r in enumerate(rows, 2):
        old = group_split.setdefault(r["sequence_id"], r["split"])
        if old != r["split"]:
            errors.append(f"row {i}: sequence {r['sequence_id']} leaks across splits")
        if r["split"] not in {"train", "val", "test"}:
            errors.append(f"row {i}: invalid split {r['split']}")
        if a.check_paths:
            for col in ("image_path", "mask_path"):
                path = Path(r[col])
                if not path.is_absolute():
                    path = a.manifest.parent.parent.parent / path
                if not path.is_file():
                    errors.append(f"row {i}: missing {col}: {path}")
            try:
                with Image.open(r["mask_path"]) as im:
                    if im.mode not in ("L", "P"):
                        errors.append(f"row {i}: mask must be single-channel indexed PNG, got {im.mode}")
                    vals = set(int(v) for v in np.asarray(im).reshape(-1))
                    bad = sorted(v for v in vals if v >= a.num_classes and v != a.ignore_index)
                    if bad:
                        errors.append(f"row {i}: unexpected mask IDs {bad[:10]}")
            except Exception as e:
                errors.append(f"row {i}: cannot read mask: {e}")
    if errors:
        print("Validation FAILED:")
        print("\n".join(errors[:100]))
        raise SystemExit(1)
    print(f"Validation passed: {len(rows)} image/mask pairs, {len(group_split)} sequence groups; no split leakage.")


if __name__ == "__main__":
    main()
