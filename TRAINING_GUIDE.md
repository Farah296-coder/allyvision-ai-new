# Dataset and Training Guide

## Scope and current readiness

This repository prepares transfer-learning workflows; it does not contain a ready-to-train annotated corpus. No full training has been run and no AllyVision-fine-tuned checkpoint exists. Existing COCO YOLO11n and Cityscapes SegFormer checkpoints remain the starting points. Training output paths are separate from `models/`.

Current inspected hardware (2026-11-21): Intel i7-1185G7, 4 cores/8 threads, 15.73 GB RAM, Intel Iris Xe, no CUDA, 22.95 GB free C:, Python 3.13.3, PyTorch 2.15.0+cpu. Use the local PC for annotation/data preparation and smoke tests. Use an NVIDIA GPU host for meaningful fine-tuning. Confirm current VRAM, disk, package/CUDA compatibility, persistence, and cloud privacy before a run.

## Taxonomies

Detection and segmentation IDs are separate namespaces. Keep labels by name when converting; never assume an ID means the same thing in both tasks.

### YOLO object detection (`datasets/config/detection_classes.yaml`)

| ID | Class | ID | Class |
|---:|---|---:|---|
| 0 | person | 10 | chair |
| 1 | car | 11 | table |
| 2 | bus | 12 | fence |
| 3 | truck | 13 | barrier |
| 4 | motorcycle | 14 | animal |
| 5 | bicycle | 15 | construction_obstacle |
| 6 | auto_rickshaw | 16 | street_vendor_cart |
| 7 | traffic_light | 17 | pothole |
| 8 | traffic_sign | 18 | rider |
| 9 | pole | 19 | other_obstacle |

Use standard YOLO detection labels: one normalized `class_id x_center y_center width height` row per box. Object boxes identify separate instances; do not annotate a broad surface region as a detection box. Define consistent box rules (visible extent, truncation, crowd/occlusion, ignore cases) before production annotation.

### SegFormer semantic segmentation (`datasets/config/segmentation_classes.yaml`)

IDs 0–17: `road`, `sidewalk`, `walkable_ground`, `crosswalk`, `curb`, `pothole`, `stairs`, `building`, `wall`, `fence`, `person`, `vehicle`, `vegetation`, `pole`, `traffic_sign`, `traffic_light`, `other`, `rider`. Pixel value 255 is ignore/void. Store masks as indexed single-channel PNG; palette-mode images must preserve pixel indices. Semantic masks label every pixel by class and do not distinguish two instances of the same class.

## Public data choices

The accompanying [dataset review](datasets/DATASET_REVIEW.md) contains the longer class, access, size, license, and conversion comparison, checked 2026-11-21. Summary:

| Source | Best contribution | Main caution | Recommendation |
|---|---|---|---|
| BDD100K | Detection, drivable area, semantic and lane context; manageable selected image/label subsets | Drivable means vehicle area, not pedestrian-safe sidewalk; use-specific restrictions | Candidate first external detector/context dataset after rights and disk check |
| IDD | Unstructured Indian urban traffic; auto-rickshaw and diverse participants | Large packages, account/edition terms; vehicle-mounted viewpoint | High-value complementary data; review exact license and select subset/storage first |
| Cityscapes | Dense urban street labels; same source domain as current SegFormer | Research/noncommercial terms; re-downloading adds little domain diversity | Keep as baseline/mapping source; do not prioritize another download |
| Mapillary Vistas | Rich global dense surface labels and varied street scenes | Version-specific size/access/license; edition must be confirmed | Valuable later segmentation complement once terms/storage are confirmed |

Dataset sizes and terms vary by edition. Never redistribute dataset files or assume one dataset's license permits commercial use. Check the current official download page and store its license/consent record alongside data provenance.

## Egyptian capture and annotation protocol

Capture from the intended device/camera mount, approximately matching camera height, pitch, field of view, resolution, and orientation. Record camera setup and sequence metadata, but avoid collecting identifying details unless needed and lawfully handled. Obtain permissions/consent appropriate to the location and subjects; set a retention and access policy before collecting footage.

Collect varied routes, times, lighting, weather, sidewalks and road crossings. Deliberately include parked and moving cars, motorcycles, tuk-tuks, people, vendors/carts, poles, barriers, uneven and blocked sidewalks, potholes, stairs, curbs, ramps, crossings, and clear walkable examples. Record continuous route/video `sequence_id`; sample representative frames rather than near-identical bursts.

Annotation rules:

1. Draw a tight box around each visible target instance; mark occlusion/truncation consistently. Do not invent a class when uncertain; use the project ignore/review process.
2. Annotate road, sidewalk, walkable ground, crosswalk, curb, pothole, stairs, and background categories per pixel using the defined taxonomy. Make boundaries consistent at curb edges, vendor carts, people, parked vehicles, and occlusions.
3. Use 255 for genuinely ambiguous/unlabeled pixels, not as a substitute for hard classes.
4. Review class counts, small objects, mask palette/index integrity, and disagreement on a shared sample before scaling up.
5. Keep annotation tasks separate for boxes and dense masks where tool/export expectations differ. CVAT supports YOLO and semantic mask export; Roboflow supports dataset versioning/export, but review private footage terms, access, and retention settings before upload.

Raw source data belongs in `datasets/raw/<source>/`; converted data in `datasets/processed/`; manifests in `datasets/manifests/`. Keep a provenance/license record per source and avoid committing large restricted data to Git.

## Split and validation requirements

Use `scripts/data/split_sequences.py` with a capture/route `sequence_id`. Never split adjacent frames from one video across train/validation/test. Use the same sequence-group split across paired images and masks. Keep an untouched held-out test set. Check class support in every split, label IDs, missing files, and duplicate/near-duplicate frames. Segmentation manifest columns: `image_path,mask_path,sequence_id,split`; required splits are `train`, `val`, `test`.

Example preparation/validation:

```powershell
.\.venv\Scripts\python.exe scripts\data\split_sequences.py capture_pairs.csv datasets\manifests\split.csv --seed 42 --train .70 --val .15
.\.venv\Scripts\python.exe scripts\data\validate_segmentation.py datasets\manifests\segmentation_manifest.csv --check-paths
```

The split script outputs grouped split assignments; arrange task files and manifests as described in `TRAINING_PREPARATION.md` and update `datasets/config/yolo11_egypt.yaml` paths. Verify configuration-relative path resolution before a real run. `--dry-run` on the training scripts checks readiness without loading weights or starting training.

## Fine-tuning procedures

### YOLO11

Starts from `models/yolo11n.pt`, never random initialization. Existing dataset YAML points at 20 project-owned classes and split directories. Run only after all three splits exist and labels match IDs:

```powershell
.\.venv\Scripts\python.exe scripts\training\train_yolo11.py --data datasets\config\yolo11_egypt.yaml --model models\yolo11n.pt --device 0 --epochs 30 --batch 8
.\.venv\Scripts\python.exe scripts\training\evaluate_yolo11.py --weights checkpoints\training\yolo11\allyvision_finetune\weights\best.pt --data datasets\config\yolo11_egypt.yaml --device 0
```

The default 640 image size and batch 8 need GPU memory; reduce batch/size or use a larger GPU if out of memory. Fine-tuned weights go to the training output directory, not over the original.

### SegFormer

Starts from the existing Cityscapes-fine-tuned model ID, reuses compatible weights, and replaces the classification head for the 18-class target taxonomy. Inputs/masks are resized and padded; current augmentation is horizontal flip. Run after the paired manifest is validated:

```powershell
.\.venv\Scripts\python.exe scripts\training\train_segformer.py --config datasets\config\segformer_dataset.yaml --epochs 10 --batch 2 --device cuda
.\.venv\Scripts\python.exe scripts\training\evaluate_segformer.py --weights checkpoints\training\segformer --config datasets\config\segformer_dataset.yaml --device cuda
```

On CUDA OOM, reduce `--batch` or `--size`; do not run full training on the inspected CPU system. Checkpoint/model access may need network or a cache. The existing training script's checkpoint argument/source behavior should be reviewed for any custom starting checkpoint before use.

## Evaluation and release gate

Report detection precision, recall, mAP50, mAP50-95 and per-class values; report segmentation mIoU and per-class IoU on the held-out test set, including class support. Also evaluate end-to-end direction outcomes on separately annotated walks; detector/segmenter metrics do not establish route safety. Include failure cases, camera viewpoint, day/night/weather, crowding, and confidence calibration analysis. Do not call training loss/accuracy evidence of navigation accuracy.

Before any release claim: legal review of all weight/data terms, reproducible environment and checkpoint provenance, sequence-isolated external test, target-device field evaluation, failure handling, and structured evaluation with visually impaired participants under an approved research protocol.

## No training has been run

No full dataset was downloaded, no dataset terms were accepted on behalf of the user, no fine-tuning was started, and no fine-tuned checkpoint exists. The next required inputs are permitted annotated data, sequence metadata/splits, a reviewed dataset license, and an adequate GPU/storage plan.
