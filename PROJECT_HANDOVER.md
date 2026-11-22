# AllyVision: Project Handover

**Status date:** 21 November 2026  
**Project type:** Experimental computer-vision navigation prototype  
**Audience:** Incoming developers, annotators, and research collaborators

> ## READ THIS BEFORE CONTINUING
> AllyVision is a research prototype, not a validated assistive navigation device. `FREE` is a heuristic classification of visible sidewalk evidence. It does not prove that a route is safe, wide enough, level, traversable, or free of hazards. Do not use it as a replacement for a white cane, guide dog, or other mobility aid. No model has yet been fine-tuned on project-specific or Egyptian street data, and no user study or real-world navigation accuracy evaluation has been completed.

## Executive summary

The Python package accepts one image, a video, or an OpenCV camera index. A pretrained SegFormer produces a semantic class map; YOLO11n detects COCO objects; an outdoor metric-depth checkpoint estimates monocular depth. `navigation.py` divides the lower frame into left, center, and right corridors and emits a conservative status and optional direction suggestion. The CLI saves JSON and a visualization.

Current live camera default is a six-panel diagnostic dashboard: original frame, segmentation overlay, object detections, metric depth, three corridors, and route decision. Image/video default to a unified overlay. These are implemented. Fine-tuning, Egyptian-specific recognition, and a validated three-panel visualization are not implemented as trained/validated product features.

## Implementation status

| Capability | Status | Evidence / limitation |
|---|---|---|
| Image, video, webcam CLI | Implemented | `allyvision/cli.py`; Windows probes DirectShow, then Media Foundation |
| Semantic segmentation | Implemented with pretrained Cityscapes SegFormer-B0 | Runtime taxonomy is upstream Cityscapes; project training taxonomy is not integrated |
| Object detection | Implemented with pretrained COCO YOLO11n | COCO vocabulary only; custom classes such as tuk-tuk, pothole, pole, vendor cart are not added by current checkpoint |
| Outdoor metric depth | Implemented with Depth Anything V2 Metric Outdoor Small | Monocular estimate; not independently calibrated for the target device |
| LEFT/CENTER/RIGHT status | Implemented heuristic | Road-only pixels cannot make a corridor FREE; thresholds are not validated safety boundaries |
| 3×3 spatial matrix | Implemented per frame | Cell-level objects, bounding-box overlap, confidence, reliable depth, walkable/road coverage, obstacle flag, and heuristic status are included in JSON and shown on the image |
| Six-panel and unified visualization | Implemented | Debug dashboard is camera default; unified overlay is image/video default |
| Project-specific fine-tuning | Not done | Training scripts are preparation only; no labeled data or trained outputs |
| Egyptian street validation | Not done | No annotated target-domain benchmark or user study |

## Model inventory

| Runtime role | Checkpoint / source | Pretraining or fine-tuning data | Local state | Notes |
|---|---|---|---|---|
| Detection | `models/yolo11n.pt` (fallback `yolo11n.pt`) | Ultralytics YOLO11n pretrained on COCO | Checkpoint in repository (two copies currently exist) | Standard 80-class COCO detector; not an arbitrary-object detector and not AllyVision-fine-tuned. Ultralytics licensing is AGPL-3.0 or Enterprise; assess before distribution. |
| Semantic segmentation | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | Cityscapes | Loaded from Hugging Face cache/network | Cityscapes labels include road and sidewalk. Upstream NVIDIA SegFormer license restricts the work/derivatives to non-commercial research/evaluation; check card and LICENSE. |
| Metric depth | `models/depth_anything_v2_metric_outdoor_small/` (fallback HF ID `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`) | Metric outdoor fine-tune; model card identifies synthetic Virtual KITTI | Local weights included | Checkpoint declares metric output and max depth 80 m. Upstream Depth Anything V2 says Small is Apache-2.0; verify exact fine-tuned checkpoint terms before redistribution/commercial use. |
| Legacy notebook depth | Referenced by `AllyVision_Live_Detection_Depth.ipynb` | Notebook says indoor-oriented; exact checkpoint/license should be verified from notebook cells before reuse | Notebook path only | Separate from outdoor metric depth used by CLI. Notebook detects and estimates depth; it does not implement the current LEFT/CENTER/RIGHT navigation pipeline. |

The three active runtime models are pretrained third-party models. **There is no AllyVision-trained checkpoint.** Fine-tuning scripts refer to these starting checkpoints and save future outputs under `checkpoints/training/`; no full training run has occurred.

## Repository map

```text
allyvision/                 Runtime pipeline, models, depth, navigation, visualizers, CLI
models/                     YOLO11n and outdoor metric-depth checkpoint
datasets/config/             Project detection/segmentation taxonomies and configs
datasets/raw/                Empty source-data destination (no public dataset downloaded)
datasets/processed/          Empty prepared-data destination
datasets/manifests/          Annotation/training manifests; current segmentation manifest has header only
scripts/data/                Sequence splitting and segmentation validation
scripts/training/            YOLO11/SegFormer fine-tuning and evaluation scripts
tests/                       Deterministic navigation and pipeline tests
outputs/                     Current generated previews/JSON; not a benchmark dataset
AllyVision_Live_Detection_Depth.ipynb  Legacy notebook workflow
```

## Install and run

Environment used for the project: Windows, Python 3.13.3, CPU PyTorch 2.15.0. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Typical use:

```powershell
.\.venv\Scripts\python.exe -m allyvision.cli --image path\to\street.jpg --no-display
.\.venv\Scripts\python.exe -m allyvision.cli --video path\to\walk.mp4 --max-frames 100
.\.venv\Scripts\python.exe -m allyvision.cli
```

The CLI prints per-frame JSON. Images are saved as `outputs/<stem>_unified.jpg` unless `--visualization debug` or `--output` is supplied. Running the CLI without a source opens a video file picker. Videos are saved as `outputs/<stem>_unified.mp4` by default. Press `q` or `Esc` or close the preview to save the frames processed so far. Add `--no-display --max-frames 1` to a video command to check the first frame. First model load can take time and may need Hugging Face network/cache access.

Relevant runtime options: `--device auto|cpu|cuda`, `--confidence`, `--imgsz`, `--segmentation-size`, `--depth-size`, `--depth-every`, `--no-depth`, `--walkable-labels`, `--road-labels`, `--visualization unified|debug`, `--output`, `--max-frames`. See `python -m allyvision.cli --help`.

## Navigation logic and visualization

The segmentation and confidence maps are resized/aligned to the input. The lower 48% of the image is split into three equal-width sectors. `FREE` requires sidewalk evidence in the sector and lower near-field band, sufficient mean confidence, and no qualifying obstacle. Current code thresholds: walkable fraction ≥0.22, bottom-band walkable fraction ≥0.18, sidewalk mean confidence ≥0.52; scene confidence below 0.48 is `UNCERTAIN`. Road-only evidence is always `UNCERTAIN`. A corridor without visible road/sidewalk may be `BLOCKED` only with sufficiently confident segmentation.

Detection boxes intersecting a corridor can be treated as hazards when they overlap substantially or span a meaningful part of it and reach the near field. With depth, an intersecting obstacle estimated at ≤2 m is `BLOCKED`; a spatially intersecting obstacle beyond that threshold but reaching the lower frame yields `UNCERTAIN`. Without depth, a qualifying near-field detection can block. In camera/video, `FREE` requires two consecutive frames; `BLOCKED` is immediate. The logic is geometric and heuristic; depth is a model estimate and obstacle vocabulary is limited to a hard-coded subset of COCO classes.

The debug dashboard has six panels. Its object-detection panel displays the aligned 3×3 cell grid; the unified view also draws the 3×3 matrix over the source frame along with semantic color, detection boxes, LEFT/CENTER/RIGHT sector status and route status. Both views update from the same per-frame inference payload. The matrix is geometric localization, not speech or tactile output. Neither visualization is an independent performance result.

The matrix JSON entry has nine row-major cells (`TOP_LEFT` through `BOTTOM_RIGHT`). Each cell carries source-frame bounds, overlapping objects, class, confidence, clipped box, center, cell-overlap fraction, reliable estimated distance (otherwise `null`), walkable and road pixel coverage, semantic confidence, obstacle presence, status and rationale. An object's full box extent is intersected with every cell; it is not assigned only to the center cell. Cell `BLOCKED` means a navigation-hazard detection overlaps it or confident segmentation finds no walkable surface; road-only cells remain `UNCERTAIN`. These status rules are heuristic and do not certify traversability.

## Dataset and training state

Project-owned class maps are in `datasets/config/`: 20 detection classes and 18 segmentation classes plus ignore ID 255. These are **future training taxonomies**, not runtime model labels. Raw and processed dataset directories do not currently contain a usable annotated training set; `datasets/manifests/segmentation_manifest.csv` has no examples. Do not assume custom labels are recognized by current pretrained weights.

`datasets/DATASET_REVIEW.md` compares BDD100K, IDD, Cityscapes, and Mapillary Vistas. High-level recommendation: collect device-view Egyptian sequences first; consider BDD100K for external detection/road context and IDD for unstructured traffic/auto-rickshaws after rights and storage review. Cityscapes is already the segmentation model's source domain. Mapillary release-specific access, size, and terms need checking. No datasets were downloaded. Review the exact license and download terms before use.

Annotation exports and sequence-safe splitting are described in `datasets/ANNOTATION_GUIDELINES.md`, `scripts/data/split_sequences.py`, and `TRAINING_GUIDE.md`. YOLO and SegFormer training/evaluation scripts are prepared but unrun. They save new checkpoints separately and preserve existing weights.

## Hardware and observed tests

Inspected system: Intel Core i7-1185G7 (4 cores/8 threads), 15.73 GB RAM, Intel Iris Xe graphics, no CUDA GPU, 22.95 GB free on C:, Python 3.13.3, PyTorch 2.15.0+cpu. Suitable for annotations, conversions, and small CPU inference checks; full fine-tuning is not practical locally. Use a suitable NVIDIA GPU host/Colab after checking VRAM, storage, and footage privacy.

The deterministic unit suite currently passes **17 tests**, including three 3×3 spatial-matrix checks, using:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The matrix was also exercised through the real CLI on `examples/street_bus.jpg` with all three runtime models enabled and on the existing short video `outputs_archive_20261121/unified_video_smoke.mp4`; the video run processed both available frames and wrote `outputs/spatial_matrix_video.mp4`. The image run wrote `outputs/spatial_matrix_street_bus.jpg` and returned nine region records with detections, cell intersections, confidence, semantic coverage, route status, and reliable depth where available. These are integration smoke checks, not accuracy evaluation. The video input is an archived generated smoke clip rather than a fresh raw webcam capture; a live camera was not tested during this matrix change.

A short webcam smoke check returned a frame through DirectShow. Earlier brief CPU timings ranged substantially by image size and model-cache state (roughly 0.2–2.4 seconds/frame in different checks); these are informal observations, not a stable benchmark. Sample-image checks produced detections on some images and none on another; all sample route statuses were `UNCERTAIN`. There is no measured navigation accuracy, mIoU, mAP on a held-out project dataset, or safety performance result.

## Dependencies and license notes

Pinned Python packages are listed in `requirements.txt` (including PyTorch, Transformers, Ultralytics, OpenCV, Pillow, NumPy, and PyYAML). Package licenses do not grant rights to model weights or datasets. Review the upstream model and dataset terms linked below. In particular, SegFormer's upstream license and Ultralytics' AGPL/commercial offering can constrain downstream use. This document is not legal advice.

## References

- [SegFormer Cityscapes model card](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024) and [upstream license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
- [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) and [Ultralytics licensing](https://www.ultralytics.com/license)
- [Depth Anything V2 Metric Outdoor Small card](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf) and [upstream project/license](https://github.com/DepthAnything/Depth-Anything-V2)
- Dataset portals: [BDD100K](https://github.com/bdd100k/bdd100k), [IDD](https://idd.insaan.iiit.ac.in/), [Cityscapes](https://www.cityscapes-dataset.com/), [Mapillary Vistas](https://www.mapillary.com/dataset/vistas)
- Annotation tools: [CVAT](https://www.cvat.ai/) and [Roboflow](https://roboflow.com/). Review current service terms and privacy settings before uploading footage.

## Handover checklist

1. Keep current pretrained checkpoints intact; put any future fine-tuned checkpoint in a new named directory.
2. Secure data-use permissions and sufficient external/cloud storage before downloading larger datasets.
3. Capture and annotate target-device Egyptian footage, split by whole route/video sequence, and inspect class balance.
4. Run dataset validators, then a small GPU pilot; evaluate on an untouched sequence-level test split.
5. Validate lane/corridor decisions separately from detector/segmenter metrics. Include diverse streets, lighting, weather, rare hazards, and user-centered evaluation before any assistive claim.
