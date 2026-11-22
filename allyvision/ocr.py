"""Read English text from video frames with Windows OCR or Tesseract."""

import base64
import csv
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2


def find_ocr_engine():
    if os.name == "nt":
        powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        if powershell.is_file():
            return "windows", str(powershell)
    executable = shutil.which("tesseract")
    if executable:
        return "tesseract", executable
    raise RuntimeError("OCR requires Windows PowerShell or Tesseract on PATH.")


def _prepare_frame(frame):
    height, width = frame.shape[:2]
    scale = min(2.0, 3000 / max(height, width))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        gray, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA,
    )
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def read_words(frame, engine):
    enhanced = _read_prepared(frame, _prepare_frame(frame), engine)
    height, width = frame.shape[:2]
    scale = min(1.0, 3000 / max(height, width))
    original = frame
    if scale < 1.0:
        original = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    words = _read_prepared(frame, original, engine)
    for entry in enhanced:
        box, _, _ = entry
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not any(left <= x <= right and top <= y <= bottom for (left, top, right, bottom), _, _ in words):
            words.append(entry)
    return words


def _read_prepared(frame, prepared, engine):
    height, width = frame.shape[:2]
    prepared_height, prepared_width = prepared.shape[:2]

    def source_box(box):
        x1, y1, x2, y2 = box
        return (
            round(x1 * width / prepared_width), round(y1 * height / prepared_height),
            round(x2 * width / prepared_width), round(y2 * height / prepared_height),
        )

    ok, encoded = cv2.imencode(".png", prepared)
    if not ok:
        raise RuntimeError("Could not encode the frame for OCR")
    backend, executable = engine
    if backend == "windows":
        script = Path(__file__).with_name("windows_ocr.ps1")
        command = [executable, "-NoProfile", "-NonInteractive", "-Command", script.read_text(encoding="utf-8")]
        data = base64.b64encode(encoded.tobytes()) + b"\n"
    else:
        command = [executable, "stdin", "stdout", "-l", "eng", "--psm", "11", "tsv"]
        data = encoded.tobytes()
    result = subprocess.run(
        command, input=data, capture_output=True, timeout=8,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip() or "OCR failed")
    if backend == "windows":
        return [(source_box(word["box"]), word["text"], None) for word in json.loads(result.stdout)]
    table = csv.DictReader(io.StringIO(result.stdout.decode("utf-8")), delimiter="\t")
    words = []
    for row in table:
        text = row["text"].strip()
        confidence = float(row["conf"])
        if not text or confidence < 55:
            continue
        x, y, width, height = (int(row[key]) for key in ("left", "top", "width", "height"))
        words.append((source_box((x, y, x + width, y + height)), text, confidence / 100))
    return words
