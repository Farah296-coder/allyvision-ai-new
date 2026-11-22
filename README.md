# AllyVision

AllyVision is an experimental street-scene perception prototype. It combines pretrained object detection, semantic segmentation, and monocular metric-depth estimation to report `LEFT`, `CENTER`, and `RIGHT` corridor statuses (`FREE`, `BLOCKED`, or `UNCERTAIN`).

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m allyvision.cli --image path\to\street.jpg --no-display
.\.venv\Scripts\python.exe -m allyvision.cli
```

Running the CLI without a source opens a file picker. AllyVision runs object detection on one of every four frames and reports progress in the terminal. Set `--detect-every 1` to analyze every frame or `--detect-every 8` for faster processing with less frequent box updates. When the annotated video is ready, it opens in the default video player on Windows at the source frame rate. The result is saved to `outputs/<video-name>_detection.mp4` and contains annotated frames without the original audio.

Use `--real-time` to play the source immediately after YOLO loads and overlay detections as they arrive. Detection runs in the background on sampled frames. Press `q` or `Esc` to close the preview. This mode does not export an MP4.

Add `--distance` to video playback or `--real-time` to show estimated camera-to-object depth in meters on the boxes. The outdoor depth model may download on first use. It estimates distance from a single frame, so the result is approximate and may be hidden when the depth inside the box is inconsistent. Enabling depth can slow detection updates on CPU.

Use `--spatial-matrix 3x3` or `--spatial-matrix 6x12` to overlay a grid of rows by columns on the detection video. In real-time playback, press `3` or `6` to switch grids.

For hardware integration, run `--real-time --matrix-json --spatial-matrix 3x3` or choose `6x12`. This enables depth automatically and fixes the grid shape for the stream. Standard output emits one JSON line per displayed frame with `frame`, `measurement_frame`, `rows`, `columns`, `unit` and `matrix`. Rows run top to bottom and columns left to right. Each detection belongs to the cell containing its box center. If multiple objects have estimates in the same cell, the matrix reports the nearest one in meters. A value of `-1` means the cell has no available estimate. Use `measurement_frame` to identify the source frame for estimates repeated across displayed frames. It is null when the last estimate has expired.

Run `--real-time --ocr` to read English text during playback. On Windows the app uses built-in OCR with an installed English recognition language. On other systems it uses Tesseract if available on PATH. Recognized words appear in yellow boxes. Combine OCR with `--distance` or the spatial matrix options as needed.

For full per-frame analysis and a unified overlay export, pass the video path with `--no-display`. Add `--visualization unified` instead to watch full analysis frame by frame.

```powershell
.\.venv\Scripts\python.exe -m allyvision.cli --video "path\to\clip.mp4" --no-display
```

In [the Colab notebook](AllyVision_Live_Detection_Depth.ipynb), run the cells in order, upload a video in step 4, and run step 8 to process the video and start the download. In local Jupyter, set `VIDEO_PATH` in step 4.

## Current models

The runtime uses pretrained checkpoints: YOLO11n (COCO), SegFormer-B0 fine-tuned on Cityscapes, and Depth Anything V2 Metric Outdoor Small. **No model has yet been fine-tuned on AllyVision or Egyptian street data.** See [PROJECT_HANDOVER.md](PROJECT_HANDOVER.md) for actual status and limitations.

The output includes a live 3×3 spatial matrix with per-cell object localization, semantic coverage, reliable distance estimates, and heuristic status. This is not a safety-certified mobility aid; statuses are not calibrated safety probabilities. Do not rely on it instead of a white cane, guide dog, or other mobility aid.

## Documentation

- [Project handover](PROJECT_HANDOVER.md): verified implementation, environment, results, risks, and references.
- [Architecture](ARCHITECTURE.md): pipeline and navigation decision flow.
- [Training guide](TRAINING_GUIDE.md): dataset readiness, annotation, fine-tuning, and evaluation.
- [Development roadmap](DEVELOPMENT_ROADMAP.md): current status and next work.
- [Dataset review](datasets/DATASET_REVIEW.md) and [annotation guidelines](datasets/ANNOTATION_GUIDELINES.md).

Run unit checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
