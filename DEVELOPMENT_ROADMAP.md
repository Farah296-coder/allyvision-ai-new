# AllyVision Development Roadmap

## Current state (21 November 2026)

| Workstream | State | Exit evidence needed |
|---|---|---|
| Runtime prototype | Implemented for image/video/webcam; includes two visualization modes | Repeatable environment and supported-camera matrix |
| Pretrained perception | COCO YOLO11n, Cityscapes SegFormer-B0, metric outdoor Depth Anything V2 Small | Checkpoint provenance, license review, target-device characterization |
| Three-way heuristic | Implemented with conservative `UNCERTAIN` default and temporal FREE hysteresis | Held-out route-level evaluation and threshold analysis |
| Dataset preparation | Taxonomies, annotation guidance, configs, splitting/validation and fine-tuning scripts prepared | Real labels, validated manifests, class balance, rights records |
| Fine-tuned models | Not started | GPU pilot, independent test metrics, new checkpoints kept separately |
| Egyptian domain adaptation | Not started | Device-viewpoint Egyptian capture and annotated evaluation |
| Assistive validation | Not started | Human factors, accessibility, failure-mode, and user evaluation |

## Recommended sequence

### 1. Freeze and document a baseline

Keep current upstream checkpoints unchanged. Record package versions, model IDs/hash, camera geometry, input settings, device, and sample JSON/image outputs. Establish a small manually labeled, sequence-separated evaluation set to quantify current failure types.

### 2. Collect target-view data

Define device mount and privacy/consent handling, then capture multiple Egyptian routes and conditions. Include sidewalks, sidewalks blocked by vendors/vehicles, curb/crosswalk/stairs/potholes, pedestrians, vehicles, tuk-tuks, poles and barriers, plus genuinely clear paths. Capture route-level metadata and split by whole sequence.

### 3. Annotate and audit

Use the separate 20-class detection and 18-class segmentation taxonomies. Pilot annotation with CVAT or an approved Roboflow workflow. Measure agreement and inspect all rare/ambiguous classes. Add BDD100K or selected IDD data only after license, viewpoint mapping, and storage decisions.

### 4. Run transfer-learning pilots

Start with a short YOLO11n fine-tune and SegFormer-B0 fine-tune on a GPU. Compare against unmodified checkpoints using a held-out test set grouped by route. Track compute, checkpoint hashes, class support, per-class metrics, and qualitative failures. Preserve original checkpoints and choose a baseline only on test evidence.

### 5. Validate route decisions

Create a dedicated route-level protocol with human labels for left/center/right traversability and hazards. Evaluate false-FREE rate, missed obstacles, coverage/UNCERTAIN rate, latency, and stability, not only component mAP/mIoU. Test day/night, rain/glare, crowds, unusual surfaces, camera motion, and calibration. Involve visually impaired users only through a properly reviewed, safe protocol.

### 6. Optimize for target hardware

Measure end-to-end latency including segmentation and depth refresh on the actual device. Consider frame resizing, asynchronous model stages, depth cadence, accelerator support, and quantization only with regression tests. Current CPU setup is suitable for prototype checks, not real-time performance claims or model training.

### 7. Resolve product and licensing questions

Review SegFormer research/evaluation terms, Ultralytics AGPL/commercial terms, Depth Anything checkpoint terms, and each dataset edition. Decide whether a commercially usable alternative is needed before distribution. Maintain a software bill of materials and model/data provenance.

## Definition of “ready for next training stage”

- Authorized, versioned datasets and documented licenses/consents.
- Nonempty train/validation/test manifests with no route/video leakage.
- Masks/boxes pass validators and annotation audit.
- Adequate persistent storage and GPU VRAM confirmed.
- Existing checkpoint backups/hashes recorded; training output path separate.
- Pre-registered evaluation metrics and independent route-level test protocol.

## Out of scope for current prototype

The current code does not map a destination, understand all traffic rules, recognize every arbitrary object, estimate traversable width or drop-offs reliably, provide audio/haptic guidance, or guarantee safe navigation. These require separate design, data, validation, and user-centered engineering.
