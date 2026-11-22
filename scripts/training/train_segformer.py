"""Transfer-learn SegFormer from the project's existing Cityscapes checkpoint."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import yaml


class MaskDataset(Dataset):
    def __init__(self, manifest, split, size, mean, std, augment=False):
        with Path(manifest).open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} examples in {manifest}")
        self.size, self.mean, self.std, self.augment = size, torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None], augment

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        image = Image.open(r["image_path"]).convert("RGB")
        with Image.open(r["mask_path"]) as src_mask:
            if src_mask.mode not in ("L", "P"):
                raise ValueError(f"Mask must be indexed grayscale PNG, got {src_mask.mode}: {r['mask_path']}")
            # Palette mode stores IDs in pixel indices; converting P->L would apply colors and corrupt IDs.
            mask = Image.fromarray(np.asarray(src_mask, dtype=np.uint8).copy(), mode="L")
        scale = self.size / max(image.size)
        wh = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(wh, Image.Resampling.BILINEAR)
        mask = mask.resize(wh, Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        mask_canvas = Image.new("L", (self.size, self.size), 255)
        offset = ((self.size-wh[0])//2, (self.size-wh[1])//2)
        canvas.paste(image, offset); mask_canvas.paste(mask, offset)
        image, mask = canvas, mask_canvas
        if self.augment and random.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.
        y = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy().astype(np.int64))
        return (x - self.mean) / self.std, y


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index):
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(pixel_values=x).logits
        pred = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        valid = y != ignore_index
        bins = torch.bincount((y[valid] * num_classes + pred[valid]), minlength=num_classes**2)
        conf += bins.reshape(num_classes, num_classes)
    tp = conf.diag().float(); denom = conf.sum(0) + conf.sum(1) - tp
    iou = tp / denom.clamp_min(1)
    present = denom > 0
    return float(iou[present].mean().item()), iou.cpu().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("datasets/config/segformer_dataset.yaml"))
    p.add_argument("--checkpoint", type=Path, default=Path("models/segformer_cityscapes"), help="local saved HF model directory")
    p.add_argument("--model-id", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024", help="existing pretrained HF checkpoint (used if local dir absent)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=Path("checkpoints/training/segformer"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.config.is_file(): p.error(f"config not found: {a.config}; copy segformer_dataset.example.yaml after preparing manifests")
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    m = Path(cfg["manifest"])
    if not m.is_absolute(): m = (a.config.parent / m).resolve()
    if not m.is_file(): p.error(f"paired image/mask manifest missing: {m}")
    with m.open(newline="", encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        if not all(k in r for k in ("image_path", "mask_path", "sequence_id", "split")):
            p.error("manifest requires image_path, mask_path, sequence_id, split")
        old = groups.setdefault(r["sequence_id"], r["split"])
        if old != r["split"]: p.error(f"sequence leakage: {r['sequence_id']}")
    if not {"train", "val", "test"}.issubset({r["split"] for r in rows}): p.error("manifest must include train, val, and test splits")
    for r in rows:
        for key in ("image_path", "mask_path"):
            if not Path(r[key]).is_file(): p.error(f"missing {key}: {r[key]}")
    if a.dry_run:
        print(f"Dry run passed: {len(rows)} pairs, {len(groups)} sequences. No weights loaded or training started.")
        return
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    names = cfg["class_names"]; n = len(names); ignore = int(cfg.get("ignore_index", 255))
    source = str(a.checkpoint) if a.checkpoint.is_dir() else a.model_id
    processor = AutoImageProcessor.from_pretrained(source)
    model = SegformerForSemanticSegmentation.from_pretrained(
        source, num_labels=n, id2label={i: v for i, v in enumerate(names)},
        label2id={v: i for i, v in enumerate(names)}, ignore_mismatched_sizes=True).to(device)
    train_ds = MaskDataset(m, "train", a.size, processor.image_mean, processor.image_std, augment=True)
    val_ds = MaskDataset(m, "val", a.size, processor.image_mean, processor.image_std)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    best = -1.
    a.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(a.epochs):
        model.train(); total = 0.
        for x, y in train_dl:
            x, y = x.to(device), y.to(device); opt.zero_grad(set_to_none=True)
            logits = model(pixel_values=x).logits
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, y, ignore_index=ignore); loss.backward(); opt.step(); total += loss.item()
        miou, per_class = evaluate(model, val_dl, device, n, ignore)
        print(f"epoch={epoch+1}/{a.epochs} train_loss={total/max(1,len(train_dl)):.4f} val_mIoU={miou:.4f}")
        if miou > best:
            best = miou; model.save_pretrained(a.output); processor.save_pretrained(a.output)
            (a.output / "validation_metrics.txt").write_text(
                f"val_mIoU={miou:.6f}\n" + "\n".join(f"{name}={v:.6f}" for name,v in zip(names,per_class)), encoding="utf-8")
    print(f"Best validation mIoU={best:.4f}. This is dataset validation, not real-world navigation accuracy.")


if __name__ == "__main__": main()
