"""Evaluate a fine-tuned YOLO checkpoint on the held-out test split."""
import argparse
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/config/yolo11_egypt.yaml"))
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()
    if not a.weights.is_file(): p.error(f"weights not found: {a.weights}")
    cfg = yaml.safe_load(a.data.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", ".")); root = root if root.is_absolute() else (a.data.parent / root).resolve()
    if not (root / cfg["test"]).exists(): p.error("test image directory is missing; do not evaluate against train/val")
    from ultralytics import YOLO
    metrics = YOLO(str(a.weights)).val(data=str(a.data), split="test", device=a.device, imgsz=a.imgsz, plots=True)
    box = metrics.box
    print(f"Precision={float(box.mp):.4f} Recall={float(box.mr):.4f} mAP50={float(box.map50):.4f} mAP50-95={float(box.map):.4f}")
    for i, name in enumerate(metrics.names.values()):
        if i < len(box.p): print(f"{name}: P={box.p[i]:.4f} R={box.r[i]:.4f} AP50={box.ap50[i]:.4f} AP50-95={box.ap[i]:.4f}")


if __name__ == "__main__": main()
