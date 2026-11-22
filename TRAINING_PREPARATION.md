# AllyVision transfer-learning preparation

## What is prepared

- YOLO11 detection taxonomy and dataset config: `datasets/config/detection_classes.yaml`, `yolo11_egypt.yaml`.
- SegFormer semantic taxonomy and paired-manifest config: `datasets/config/segmentation_classes.yaml`, `segformer_dataset.yaml`.
- Dataset name mappings, capture metadata example, annotation policy, empty raw/processed destinations, and review of public sources: `datasets/`.
- Deterministic sequence-level splitter: `scripts/data/split_sequences.py`.
- Mask/manifest validator: `scripts/data/validate_segmentation.py`.
- YOLO11 fine-tune and held-out test evaluator: `scripts/training/train_yolo11.py`, `evaluate_yolo11.py`.
- SegFormer transfer-learning loop with best-checkpoint-on-validation-mIoU saving and held-out test evaluator: `scripts/training/train_segformer.py`, `evaluate_segformer.py`.

The YOLO training script defaults to the existing `models/yolo11n.pt`; it refuses to proceed if the checkpoint is absent. SegFormer defaults to the existing Cityscapes fine-tuned model ID from the current implementation. Fine-tuned checkpoints are written to `checkpoints/training/`, not over existing model files. SegFormer replaces its output classifier because the target project taxonomy differs; the pretrained encoder/decoder weights are reused where shapes match (`ignore_mismatched_sizes=True`). The script downloads no data itself, though Transformers may fetch the named checkpoint if not cached. Network/model access must be available to start a real training run.

## Add data when ready

1. Obtain and review the source data terms and access approval; recheck disk space. Keep source data under the matching `datasets/raw/<source>/` folder (or an external storage path).
2. Annotate Egyptian capture sequences in separate CVAT/Roboflow projects for boxes and semantic masks. Export to the formats in `datasets/ANNOTATION_GUIDELINES.md`.
3. Build one CSV per task with `image_path,sequence_id` (detection) or `image_path,mask_path,sequence_id` (segmentation). Use a sequence ID that groups all frames from one recording/route capture, not one ID per frame.
4. From the repository root, create deterministic splits. Example:

```powershell
.\.venv\Scripts\python.exe scripts\data\split_sequences.py capture_pairs.csv datasets\manifests\split.csv --seed 42 --train .70 --val .15
```

For detection, convert/sort files into `datasets/processed/yolo_detection/{images,labels}/{train,val,test}` and edit `datasets/config/yolo11_egypt.yaml`. Ensure train/val/test groups came from the same sequence split. For segmentation, copy split rows into `datasets/manifests/segmentation_manifest.csv` with columns `image_path,mask_path,sequence_id,split`; masks must be single-channel indexed PNGs whose values match IDs 0–17 and void 255. Then validate:

```powershell
.\.venv\Scripts\python.exe scripts\data\validate_segmentation.py datasets\manifests\segmentation_manifest.csv --check-paths
```

5. Review licenses, class counts and group separation before training. Tiny datasets and unrepresented hazard classes are not suitable for a reliability claim.

## Training (not run)

After data, annotations, terms and compute are checked, start with a short GPU pilot:

```powershell
.\.venv\Scripts\python.exe scripts\training\train_yolo11.py --data datasets\config\yolo11_egypt.yaml --model models\yolo11n.pt --device 0 --epochs 30 --batch 8
.\.venv\Scripts\python.exe scripts\training\evaluate_yolo11.py --weights checkpoints\training\yolo11\allyvision_finetune\weights\best.pt --data datasets\config\yolo11_egypt.yaml --device 0
.\.venv\Scripts\python.exe scripts\training\train_segformer.py --config datasets\config\segformer_dataset.yaml --epochs 10 --batch 2 --device cuda
.\.venv\Scripts\python.exe scripts\training\evaluate_segformer.py --weights checkpoints\training\segformer --config datasets\config\segformer_dataset.yaml --device cuda
```

`--dry-run` checks the YOLO/SegFormer input readiness without loading weights or training. Use `--device cpu` only for a tiny code smoke-run; full training on this machine is not practical. Existing checkpoint files remain untouched.

## Metrics and interpretation

- Detection test metrics: precision, recall, mAP@0.5, mAP@0.5:0.95, plus per-class precision/recall/AP.
- Segmentation: test mean IoU and per-class IoU; validation mIoU chooses the saved checkpoint.
- Preserve an independent test split by capture sequence and do not tune against it. Report class support and source-domain breakdown. Do not call these training metrics navigation accuracy or safety evidence. Validate route decisions on held-out Egyptian walks, camera placement, night/weather, rare hazards, and with visually impaired users before any assistive claim.

## Hardware decision (2026-11-21)

Inspected device: Intel Core i7-1185G7 (4C/8T), 15.73 GB RAM, Intel Iris Xe (no CUDA), project Python 3.13.3, PyTorch 2.15.0+cpu, 0 CUDA GPUs, and 22.95 GB free on C:. This is appropriate for annotation, dataset conversion, inference and small pipeline smoke tests. Use Google Colab or another NVIDIA GPU host for actual fine-tuning; choose image/batch size after checking available VRAM. Keep training data and checkpoints on persistent storage, split sequences before upload, and review cloud privacy/retention terms for Egyptian footage. Full IDD/Mapillary data will not fit comfortably on the current free space.

No training was started because no target annotations or approved dataset are present, dataset terms have not been accepted for this use, and the local machine has no CUDA GPU.
