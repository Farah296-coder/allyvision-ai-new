# Dataset preparation (no datasets downloaded)

This directory contains schemas and empty destinations only. Never place a dataset here until its exact version, license, consent/privacy basis, and disk requirement have been reviewed. Keep raw source data read-only where possible and store derived/converted copies under `processed/`.

## Layout

```text
datasets/
  raw/{egyptian,bdd100k,idd,cityscapes,mapillary_vistas}/
  processed/{yolo_detection,segmentation}/
  manifests/                 # private capture/split manifests; gitignored
  config/                    # project class taxonomies and dataset configs
```

## Priority

1. Collect consented Egyptian, device-viewpoint street sequences. These are the essential target-domain data for sidewalk obstructions, tuk-tuks, vendors, uneven paving, curbs, stairs and potholes.
2. Evaluate BDD100K (small key-frame package, detection and drivable-area/segmentation labels) for transfer and pipeline testing. It is US dash-camera data, not pedestrian-camera safety data.
3. IDD is the strongest listed appearance complement for informal, unstructured traffic and auto-rickshaws, but its useful detection and segmentation packages are very large. Start only with IDD Lite or a licensed small subset after verifying access and storage.
4. Cityscapes is class-compatible for urban semantic segmentation but redundant with the current Cityscapes fine-tuned SegFormer checkpoint and has registered-download/custom non-commercial terms. Use for mapping/benchmarking only after terms review.
5. Mapillary Vistas has rich street-level semantic and instance annotations but version-dependent labels, large archives, account/download gating, and a separate research license. Check the exact edition and terms before acquisition.

## Storage on inspected machine

At preparation time, C: had 22.95 GB free; system RAM was 15.73 GB; CPU was Intel i7-1185G7 (4 cores/8 threads); GPU was Intel Iris Xe and PyTorch reported CPU-only, CUDA unavailable. This is insufficient for IDD Detection (22.8 GB) plus extraction, or IDD segmentation (18.5 GB) plus processing copies. BDD100K's 5.3 GB 100K image archive may fit alone but extracted images, labels, caches, checkpoints and free-space headroom make it a tight local plan. Do not download until space is rechecked. Prefer cloud storage/GPU for full training.

## Data formats

- Detection: YOLO text files, one `class_id x_center y_center width height` row per box; normalized coordinates, zero-based class IDs. See `config/detection_classes.yaml`.
- Optional instance polygons: YOLO segmentation row `class_id x1 y1 ... xn yn`, normalized; use only when instance boundaries are needed. It is distinct from semantic masks.
- Segmentation: paired RGB image and single-channel PNG mask. Pixel value is the zero-based target ID from `config/segmentation_classes.yaml`; `255` is ignore/void. Convert source labels by class name, not by assuming source IDs align.
- Train/validation/test split by capture sequence/video, never by adjacent frames. `scripts/data/split_sequences.py` creates a deterministic sequence-level manifest.

See `ANNOTATION_GUIDELINES.md`, `config/source_mappings.yaml`, and `config/egyptian_capture_manifest.example.csv`.
