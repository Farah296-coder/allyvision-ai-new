"""Command line entry point for images and video files."""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .navigation import TemporalStatusSmoother, estimate_box_distance
from .ocr import find_ocr_engine, read_words
from .pipeline import WINDOW_NAME, process_frame
from .spatial_matrix import build_distance_matrix, draw_grid_lines, render_distance_matrix


VIDEO_GRIDS = {"3x3": (3, 3), "6x12": (6, 12)}


def _fit_preview_to_screen(image):
    """Scale a dashboard to fit the visible desktop without changing saved output."""
    max_width, max_height = 1440, 810
    if os.name == "nt":
        try:
            import ctypes
            max_width = min(max_width, max(640, ctypes.windll.user32.GetSystemMetrics(0) - 80))
            max_height = min(max_height, max(360, ctypes.windll.user32.GetSystemMetrics(1) - 140))
        except (AttributeError, OSError):
            pass
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def select_video_file():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        filename = filedialog.askopenfilename(
            parent=root,
            title="Choose a video for AllyVision",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return Path(filename) if filename else None


def parse_args():
    parser = argparse.ArgumentParser(description="AllyVision three-direction navigation prototype")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Analyze one image")
    source.add_argument("--video", type=Path, help="Analyze a video file")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--confidence", type=float,
                        help="Minimum YOLO confidence (0.25 in real time/full analysis, 0.35 for export)")
    parser.add_argument("--imgsz", type=int,
                        help="YOLO input size (640 for video detection, 416 for full analysis)")
    parser.add_argument("--segmentation-size", type=int, default=256, help="SegFormer input edge size; larger can improve detail but runs slower")
    parser.add_argument("--depth-every", type=int, default=5, help="Run depth every N video frames (image mode always runs it)")
    parser.add_argument("--depth-size", type=int, default=336, help="Metric depth input edge size, at least 224 and divisible by 14")
    parser.add_argument("--no-depth", action="store_true", help="Skip metric depth; obstacle-distance output is unavailable")
    parser.add_argument("--walkable-labels", default="sidewalk", help="Comma-separated SegFormer classes allowed as walkable (e.g. sidewalk)")
    parser.add_argument("--road-labels", default="road", help="Comma-separated road classes, which remain UNCERTAIN by default")
    parser.add_argument("--visualization", choices=("unified", "debug"), default=None,
                        help="Run full per-frame analysis with a unified overlay or debug dashboard")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU inference threads; one thread was fastest on the tested CPU")
    parser.add_argument("--output", type=Path, help="Output image/video path")
    parser.add_argument("--no-display", action="store_true", help="Do not open a preview window")
    parser.add_argument("--max-frames", type=int, help="Optional cap for video processing")
    parser.add_argument("--detect-every", type=int, default=4,
                        help="Analyze one frame every N frames in video playback mode")
    parser.add_argument("--real-time", action="store_true",
                        help="Play while detecting in the background; skip MP4 export")
    parser.add_argument("--distance", action="store_true",
                        help="Estimate camera-to-object depth in meters on video detection boxes")
    parser.add_argument("--spatial-matrix", choices=VIDEO_GRIDS,
                        help="Overlay a grid of rows by columns on the detection video")
    parser.add_argument("--matrix-json", action="store_true",
                        help="Stream numeric distance matrices as JSON lines during real-time playback")
    parser.add_argument("--ocr", action="store_true", help="Read English text in real-time video playback")
    return parser.parse_args()


def _draw_detection(frame, box, label, confidence, distance_m=None, *, color=(0, 220, 0)):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(round, box)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.72, width / 1200, height / 675)
    thickness = max(1, round(scale * 1.7))
    padding = thickness + 3
    text = label if confidence is None else f"{label} {confidence:.0%}"
    if distance_m is not None:
        text += f" ~{distance_m:.1f} m"
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_width = text_width + 2 * padding
    label_height = text_height + baseline + 2 * padding
    left = min(max(x1, 0), max(0, width - label_width))
    top = y1 - label_height if y1 >= label_height else y1
    top = min(max(top, 0), max(0, height - label_height))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.rectangle(frame, (left, top), (left + label_width, top + label_height), (20, 55, 20), cv2.FILLED)
    cv2.putText(frame, text, (left + padding, top + padding + text_height),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _box_overlap(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if not intersection:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _stabilize_detections(detections, previous, confirm_hits):
    unmatched = list(previous)
    tracks = []
    for box, label, confidence, distance_m in sorted(detections, key=lambda item: item[2], reverse=True):
        index, overlap = max(
            ((i, _box_overlap(box, track[0])) for i, track in enumerate(unmatched) if track[1] == label),
            key=lambda match: match[1],
            default=(-1, 0.0),
        )
        if overlap >= 0.3:
            old_box, _, old_confidence, hits, _, _ = unmatched.pop(index)
            box = [0.65 * current + 0.35 * old for current, old in zip(box, old_box)]
            confidence = 0.65 * confidence + 0.35 * old_confidence
            tracks.append((box, label, confidence, hits + 1, 0, distance_m))
        else:
            tracks.append((box, label, confidence, 1, 0, distance_m))
    # Keep confirmed boxes through a brief missed detection to avoid flicker.
    tracks.extend((box, label, confidence, hits, misses + 1, None)
                  for box, label, confidence, hits, misses, _ in unmatched
                  if hits >= confirm_hits and misses < 2)
    return tracks


def _confirmed_words(current, recent_words):
    return [
        (box, word, confidence)
        for box, word, confidence in current
        if any(character.isalnum() for character in word) and any(
            word.casefold() == old_word.casefold()
            and _box_overlap(box, old_box) >= 0.10
            for sample in recent_words
            for old_box, old_word, _ in sample
        )
    ]


def _video_detector(args):
    import torch
    from ultralytics import YOLO

    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        torch.set_num_threads(max(1, min(args.cpu_threads, torch.get_num_threads())))
    local_weights = Path(__file__).resolve().parent.parent / "yolo11n.pt"
    model = YOLO(str(local_weights) if local_weights.is_file() else "yolo11n.pt")
    depth_model = None
    if args.distance:
        from .depth import MetricOutdoorDepth

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        depth_model = MetricOutdoorDepth(device, input_size=args.depth_size)

    def predict(frame):
        result = model.predict(
            frame, conf=args.confidence, imgsz=args.imgsz, verbose=False,
            device=None if args.device == "auto" else args.device,
        )[0]
        boxes = [] if result.boxes is None else result.boxes.data.cpu().tolist()
        if depth_model is None or not boxes:
            return [(box[:4], result.names[int(box[-1])], box[-2], None) for box in boxes]

        from PIL import Image

        depth_map = depth_model.estimate(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        detections = []
        for box in boxes:
            distance_m, spread = estimate_box_distance(depth_map, box[:4])
            if spread is None or spread > 0.35 or distance_m >= depth_model.max_depth_m:
                distance_m = None
            detections.append((box[:4], result.names[int(box[-1])], box[-2], distance_m))
        return detections

    return predict


def play_real_time(args):
    ocr_engine = find_ocr_engine() if args.ocr else None
    capture = cv2.VideoCapture(str(args.video))
    window_created = False
    frame_index = 0
    last_detection = 0
    confirm_hits = 2
    grid = VIDEO_GRIDS.get(args.spatial_matrix)
    executor = ThreadPoolExecutor(max_workers=2 if ocr_engine else 1)
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"No readable frames in {args.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 20.0
        print("Loading YOLO and detecting objects in the first frame...", file=sys.stderr, flush=True)
        predict = executor.submit(_video_detector, args).result()
        first_detections = executor.submit(predict, frame).result()
        tracks = _stabilize_detections(first_detections, [], confirm_hits)
        recent_words = deque(maxlen=2)
        words = []
        if ocr_engine:
            words = read_words(frame, ocr_engine)
            recent_words.append(words)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_created = True
        matrix_window = None
        if args.matrix_json:
            matrix_window = f"AllyVision Spatial Matrix {args.spatial_matrix} (m)"
            cv2.namedWindow(matrix_window, cv2.WINDOW_AUTOSIZE)
        started = time.perf_counter()
        result_time = started
        measured_frame = 1
        pending = None
        ocr_pending = None
        next_ocr = started + 0.15
        ocr_expires = started + 2.0
        confirmed_words = []
        while ok:
            now = time.perf_counter()
            if ocr_pending is not None and ocr_pending.done():
                current_words = ocr_pending.result()
                confirmed = _confirmed_words(current_words, recent_words)
                readable = [entry for entry in current_words if any(ch.isalnum() for ch in entry[1])]
                if readable:
                    words = readable
                    confirmed_words = confirmed
                    ocr_expires = now + 2.0
                recent_words.append(current_words)
                ocr_pending = None
            if ocr_engine and ocr_pending is None and now >= next_ocr:
                ocr_pending = executor.submit(read_words, frame.copy(), ocr_engine)
                next_ocr = now + 0.25
            if pending is not None and pending.done():
                tracks = _stabilize_detections(pending.result(), tracks, confirm_hits)
                result_time = now
                measured_frame = last_detection + 1
                pending = None
            if pending is None and frame_index - last_detection >= args.detect_every:
                pending = executor.submit(predict, frame.copy())
                last_detection = frame_index
            if grid is not None:
                draw_grid_lines(frame, *grid)
            measured_objects = []
            if now - result_time <= 0.5:
                for box, label, confidence, hits, misses, distance_m in tracks:
                    if misses == 0 and (hits >= confirm_hits or confidence >= 0.70):
                        _draw_detection(frame, box, label, confidence, distance_m)
                        measured_objects.append((box, distance_m))
            if args.matrix_json:
                rows, columns = grid
                matrix = build_distance_matrix(frame.shape, rows, columns, measured_objects)
                print(json.dumps({"frame": frame_index + 1, "rows": rows, "columns": columns,
                                  "measurement_frame": measured_frame if now - result_time <= 0.5 else None,
                                  "unit": "m", "matrix": matrix}, separators=(",", ":")), flush=True)
                cv2.imshow(matrix_window, render_distance_matrix(matrix))
            if now < ocr_expires:
                for box, word, confidence in words:
                    _draw_detection(frame, box, word, confidence, color=(0, 220, 255))
            shown = _fit_preview_to_screen(frame)
            if ocr_engine:
                status = "OCR: looking for English text"
                if words and now < ocr_expires:
                    status = "OCR: text confirmed" if confirmed_words else "OCR: first reading"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), _ = cv2.getTextSize(status, font, 0.55, 1)
                y = shown.shape[0] - 10
                cv2.rectangle(shown, (5, y - text_height - 5), (text_width + 13, y + 5), (25, 25, 25), cv2.FILLED)
                cv2.putText(shown, status, (9, y), font, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
            if frame_index == 0:
                cv2.resizeWindow(WINDOW_NAME, shown.shape[1], shown.shape[0])
            cv2.imshow(WINDOW_NAME, shown)
            frame_index += 1
            deadline = started + frame_index / fps
            delay = max(1, round(1000 * (deadline - time.perf_counter())))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                break
            if not args.matrix_json:
                if key == ord("3"):
                    grid = VIDEO_GRIDS["3x3"]
                elif key == ord("6"):
                    grid = VIDEO_GRIDS["6x12"]
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break
            ok, frame = capture.read()
        if pending is not None and pending.done():
            pending.result()
    finally:
        capture.release()
        if window_created:
            cv2.destroyAllWindows()
        executor.shutdown(cancel_futures=True)


def play_video(args):
    """Export frame-aligned YOLO detections, then play the finished video."""
    capture = cv2.VideoCapture(str(args.video))
    writer = None
    frame_count = 0
    detected_count = 0
    tracks = []
    confirm_hits = 2
    grid = VIDEO_GRIDS.get(args.spatial_matrix)
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"No readable frames in {args.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 20.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.max_frames and total_frames > args.max_frames:
            total_frames = args.max_frames
        print("Preparing annotated video. Playback starts after processing.", flush=True)

        predict = _video_detector(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video: {args.output}")

        while ok:
            if frame_count % args.detect_every == 0:
                tracks = _stabilize_detections(predict(frame), tracks, confirm_hits)
            if grid is not None:
                draw_grid_lines(frame, *grid)
            for box, label, confidence, hits, _, distance_m in tracks:
                if hits >= confirm_hits:
                    _draw_detection(frame, box, label, confidence, distance_m)
                    detected_count += 1
            writer.write(frame)
            frame_count += 1
            if frame_count == 1 or frame_count % 30 == 0:
                progress = f"/{total_frames}" if total_frames > 0 else ""
                print(f"Processed {frame_count}{progress} frames", flush=True)
            if args.max_frames and frame_count >= args.max_frames:
                break
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    print(f"Saved {frame_count} frames with {detected_count} detections: {args.output.resolve()}", flush=True)
    if os.name == "nt":
        try:
            os.startfile(str(args.output.resolve()))
        except OSError as exc:
            print(f"Could not open the default video player: {exc}. Open the saved MP4 manually.")
    else:
        preview_capture = cv2.VideoCapture(str(args.output))
        try:
            while True:
                ok, frame = preview_capture.read()
                if not ok:
                    break
                cv2.imshow(WINDOW_NAME, _fit_preview_to_screen(frame))
                if cv2.waitKey(max(1, round(1000 / fps))) & 0xFF in (ord("q"), 27):
                    break
        finally:
            preview_capture.release()
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    visualization = args.visualization or "unified"
    detection_playback = args.image is None and not args.no_display and args.visualization is None
    if args.ocr and not args.real_time:
        raise ValueError("--ocr requires --real-time")
    if args.matrix_json:
        if not args.real_time or args.spatial_matrix is None:
            raise ValueError("--matrix-json requires --real-time and --spatial-matrix")
        args.distance = True
    if args.spatial_matrix is not None and not detection_playback:
        raise ValueError("--spatial-matrix requires video detection playback")
    if args.distance and (not detection_playback or args.no_depth):
        raise ValueError("--distance requires video detection playback and depth enabled")
    if args.real_time and (not detection_playback or args.output is not None):
        raise ValueError("--real-time requires video playback without --output, --no-display or --visualization")
    if args.confidence is None:
        if args.real_time:
            args.confidence = 0.25
        else:
            args.confidence = 0.35 if detection_playback else 0.25
    if args.imgsz is None:
        args.imgsz = 640 if detection_playback else 416
    if args.image is None:
        if args.video is None:
            args.video = select_video_file()
            if args.video is None:
                return
        if not args.video.is_file():
            raise FileNotFoundError(f"Could not find video: {args.video}")
        if detection_playback and args.detect_every < 1:
            raise ValueError("--detect-every must be at least 1")
        if args.real_time:
            play_real_time(args)
            return
        suffix = "detection" if detection_playback else visualization
        args.output = args.output or Path("outputs") / f"{args.video.stem}_{suffix}.mp4"
        if args.video.resolve() == args.output.resolve() or (
            args.output.exists() and args.video.samefile(args.output)
        ):
            raise ValueError("The output path must be different from the input video.")
        if detection_playback:
            play_video(args)
            return
    if args.depth_every < 1:
        raise ValueError("--depth-every must be at least 1")
    from .models import PerceptionModels

    models = PerceptionModels(
        args.device, segmentation_size=args.segmentation_size, cpu_threads=args.cpu_threads,
        enable_depth=not args.no_depth,
        walkable_labels=tuple(args.walkable_labels.split(",")), road_labels=tuple(args.road_labels.split(",")),
        depth_size=args.depth_size,
    )
    depth_status = models.depth_model.max_depth_m if models.depth_model else None
    print(f"Loaded SegFormer, YOLO11n and metric outdoor depth={depth_status} m on {models.device}; visualization={visualization}; sidewalk IDs={sorted(models.walkable_ids)}; road IDs={sorted(models.road_ids)}")

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            # OpenCV builds on Windows may not include AVIF support; Pillow does.
            from PIL import Image
            try:
                rgb = Image.open(args.image).convert("RGB")
                frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                raise FileNotFoundError(f"Could not read image: {args.image} ({exc})") from exc
        started = time.perf_counter()
        payload, preview, _ = process_frame(
            frame, models, detector_confidence=args.confidence, imgsz=args.imgsz,
            visualization=visualization,
        )
        payload["inference_seconds"] = round(time.perf_counter() - started, 3)
        payload["source"] = str(args.image)
        print(json.dumps(payload, indent=2))
        suffix = "_unified" if visualization == "unified" else "_debug"
        output = args.output or Path("outputs") / f"{args.image.stem}{suffix}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), preview)
        print(f"Saved visualization: {output.resolve()}")
        if not args.no_display:
            shown = _fit_preview_to_screen(preview)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, shown.shape[1], shown.shape[0])
            cv2.imshow(WINDOW_NAME, shown)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video: {args.video}")
    window_name = WINDOW_NAME
    writer = None
    frame_count = 0
    cached_depth = None
    smoother = TemporalStatusSmoother(free_confirm_frames=2)
    display_window_created = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if frame_count == 0:
                    raise RuntimeError(f"The video has no readable frames: {args.video}")
                break
            frame_started = time.perf_counter()
            frame_count += 1
            if not args.no_depth and (cached_depth is None or (frame_count - 1) % args.depth_every == 0):
                from PIL import Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cached_depth = models.estimate_depth(Image.fromarray(rgb))
            payload, preview, _ = process_frame(
                frame, models, detector_confidence=args.confidence, imgsz=args.imgsz,
                depth_map_m=cached_depth, run_depth=False, smoother=smoother,
                visualization=visualization,
            )
            payload["inference_seconds"] = round(time.perf_counter() - frame_started, 3)
            print(json.dumps({"frame": frame_count, **payload}, separators=(",", ":")))
            if writer is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                preview_h, preview_w = preview.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or fps <= 0:
                    fps = 20.0
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (preview_w, preview_h))
                if not writer.isOpened():
                    raise RuntimeError(f"Unable to create output video: {args.output}")
            writer.write(preview)
            if not args.no_display:
                shown = _fit_preview_to_screen(preview)
                if not display_window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, shown.shape[1], shown.shape[0])
                    display_window_created = True
                cv2.imshow(window_name, shown)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("Video processing interrupted.")
    finally:
        capture.release()
        if writer:
            writer.release()
        if display_window_created:
            cv2.destroyAllWindows()
    if frame_count and writer is not None:
        print(f"Saved annotated video ({frame_count} frames): {args.output.resolve()}")


if __name__ == "__main__":
    main()

