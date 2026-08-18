#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import cv2 as cv
import numpy as np


def emit(obj):
    print(json.dumps(obj), flush=True)


def load_fire_module(script_path):
    spec = importlib.util.spec_from_file_location("atomic_fire_detector", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load fire detector script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_box(box, width, height):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def save_preview(path, frame):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.jpg"
    if cv.imwrite(tmp, frame, [cv.IMWRITE_JPEG_QUALITY, 82]):
        os.replace(tmp, path)


def save_fire_crop(frame, box, events_dir, stamp, index):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = safe_box(box, w, h)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * 0.15)
    pad_y = int(bh * 0.15)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    name = f"fire_crop_{stamp}_{index}.jpg"
    out = os.path.join(events_dir, name)
    if not cv.imwrite(out, crop, [cv.IMWRITE_JPEG_QUALITY, 92]):
        return None
    return name


def main():
    ap = argparse.ArgumentParser(description="Atomic Vision fire-demo event worker")
    ap.add_argument("--input", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--preview", default="")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--nms-threshold", type=float, default=0.45)
    ap.add_argument("--sample-fps", type=float, default=6.0)
    ap.add_argument("--event-cooldown", type=float, default=2.0)
    args = ap.parse_args()

    for label, p in (("input", args.input), ("script", args.script), ("model", args.model), ("library", args.library)):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{label} file not found: {p}")
    os.makedirs(args.events_dir, exist_ok=True)

    fire = load_fire_module(args.script)
    fire.OBJ_THRESH = float(args.threshold)
    fire.NMS_THRESH = float(args.nms_threshold)
    fire.CLASSES = ("fire", "other", "smoke")
    fire.NUM_CLS = 3
    fire.LISTSIZE = 67

    detector = fire.DetectionWorker(
        args.model,
        args.library,
        process_interval=0.0,
        debug=False,
        result_log_every=999999,
    )
    detector.initialize()

    cap = cv.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = float(cap.get(cv.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(fps / max(0.1, args.sample_fps))))
    frame_index = 0
    processed = 0
    last_fire_event_at = -1e9

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            current_index = frame_index
            frame_index += 1
            if current_index % frame_step != 0:
                continue

            processed += 1
            _seq, boxes, scores, classes = detector.process_frame_sync(current_index + 1, frame)
            video_time = current_index / fps

            annotated = frame.copy()
            fire_hits = []
            for box, score, cls_id in zip(boxes, scores, classes):
                cls_id = int(cls_id)
                label = fire.CLASSES[cls_id] if 0 <= cls_id < len(fire.CLASSES) else str(cls_id)
                if label not in ("fire", "smoke"):
                    continue
                x1, y1, x2, y2 = safe_box(box, annotated.shape[1], annotated.shape[0])
                color = fire.CLASS_COLORS.get(label, (0, 0, 255))
                cv.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                cv.putText(
                    annotated,
                    f"{label.upper()} {float(score):.2f}",
                    (x1, max(25, y1 - 8)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                )
                if label == "fire":
                    fire_hits.append((box, float(score), cls_id))

            # Live UI preview: always write the currently processed frame, with
            # boxes when fire/smoke is detected. Atomic replace avoids partial JPGs.
            save_preview(args.preview, annotated)

            if total_frames > 0:
                emit({"type": "progress", "progress": min(99, int((current_index + 1) * 100 / total_frames))})

            # Events are FIRE ONLY, and each saved image is only the detected
            # fire region (with a small context pad), never the full video frame.
            if fire_hits and video_time - last_fire_event_at >= args.event_cooldown:
                now = datetime.now(timezone.utc)
                stamp = now.strftime("%Y%m%d_%H%M%S_%f")
                for idx, (box, score, cls_id) in enumerate(fire_hits):
                    image_name = save_fire_crop(frame, box, args.events_dir, stamp, idx)
                    if not image_name:
                        continue
                    emit({
                        "type": "event",
                        "id": f"fire_{stamp}_{idx}",
                        "label": "fire",
                        "score": score,
                        "timestamp": now.isoformat(),
                        "video_time": video_time,
                        "image": image_name,
                    })
                last_fire_event_at = video_time
    finally:
        cap.release()
        detector.stop()

    emit({"type": "progress", "progress": 100})
    emit({"type": "done", "processed_frames": processed})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FIRE DEMO ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
