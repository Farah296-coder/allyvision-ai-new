"""Fine-tune an existing YOLO11 checkpoint; this never starts from random weights."""
import argparse
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("datasets/config/yolo11_egypt.yaml"))
    p.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"), help="existing pretrained checkpoint")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0", help="CUDA device; use cpu only for smoke/debug, not practical full training")
    p.add_argument("--project", type=Path, default=Path("checkpoints/training/yolo11"))
    p.add_argument("--name", default="allyvision_finetune")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.model.is_file():
        p.error(f"pretrained checkpoint not found: {a.model}; refusing scratch training")
    if not a.data.is_file():
        p.error(f"dataset YAML not found: {a.data}")
    cfg = yaml.safe_load(a.data.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (a.data.parent / root).resolve()
    missing = [str(root / cfg[k]) for k in ("train", "val", "test") if not (root / cfg[k]).exists()]
    if missing:
        p.error("dataset splits are not ready; missing: " + ", ".join(missing))
    print(f"Checkpoint: {a.model}; classes: {len(cfg['names'])}; dataset: {a.data}")
    if a.dry_run:
        print("Dry run passed. No model loaded and no training started.")
        return
    from ultralytics import YOLO
    model = YOLO(str(a.model))
    model.train(data=str(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=a.device, project=str(a.project), name=a.name, pretrained=True,
                resume=False, seed=42, deterministic=True, workers=2, cache=False,
                close_mosaic=10, patience=10, save=True, val=True)


if __name__ == "__main__":
    main()
