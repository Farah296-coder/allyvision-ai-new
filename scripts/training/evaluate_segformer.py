"""Report test-set mIoU and per-class IoU for an indexed-mask manifest."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("datasets/config/segformer_dataset.yaml"))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args(); cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    m = Path(cfg["manifest"]); m = m if m.is_absolute() else (a.config.parent / m).resolve()
    with m.open(newline="", encoding="utf-8-sig") as f: rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if not rows: p.error("no held-out test rows in manifest")
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(a.weights).to(device).eval()
    n, ignore = int(cfg["num_labels"]), int(cfg.get("ignore_index",255)); conf=torch.zeros((n,n),dtype=torch.int64,device=device)
    mean=torch.tensor([.485,.456,.406])[:,None,None]; std=torch.tensor([.229,.224,.225])[:,None,None]
    with torch.inference_mode():
        for r in rows:
            im=Image.open(r["image_path"]).convert("RGB")
            with Image.open(r["mask_path"]) as sm:
                if sm.mode not in ("L","P"): raise ValueError(f"Mask must be indexed grayscale PNG: {r['mask_path']}")
                mask=Image.fromarray(np.asarray(sm,dtype=np.uint8).copy(),mode="L")
            scale=a.size/max(im.size); wh=(max(1,round(im.width*scale)),max(1,round(im.height*scale)))
            im=im.resize(wh,Image.Resampling.BILINEAR); mask=mask.resize(wh,Image.Resampling.NEAREST)
            canvas=Image.new("RGB",(a.size,a.size),(114,114,114)); mc=Image.new("L",(a.size,a.size),ignore)
            off=((a.size-wh[0])//2,(a.size-wh[1])//2); canvas.paste(im,off); mc.paste(mask,off); im,mask=canvas,mc
            x=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()).permute(2,0,1)/255.; x=((x-mean)/std)[None].to(device)
            y=torch.from_numpy(np.asarray(mask,dtype=np.uint8).copy().astype(np.int64)).to(device)
            pred=F.interpolate(model(pixel_values=x).logits,size=y.shape,mode="bilinear",align_corners=False).argmax(1)[0]
            valid=y!=ignore; conf+=torch.bincount(y[valid]*n+pred[valid],minlength=n*n).reshape(n,n)
    tp=conf.diag().float(); denom=conf.sum(0)+conf.sum(1)-tp; iou=tp/denom.clamp_min(1); present=denom>0
    print(f"Test mIoU={float(iou[present].mean()):.4f} (dataset metric; not real-world navigation accuracy)")
    for name,val,seen in zip(cfg["class_names"],iou.cpu().tolist(),present.cpu().tolist()): print(f"{name}: {val:.4f}" if seen else f"{name}: n/a (absent in test labels)")


if __name__ == "__main__": main()
