#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import time
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
    ap = argparse.ArgumentParser(description="Atomic Vision real-time fire demo worker")
    ap.add_argument("--input", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--nms-threshold", type=float, default=0.45)
    ap.add_argument("--sample-fps", type=float, default=4.0)
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
    duration = (total_frames / fps) if total_frames > 0 else 0.0
    interval = 1.0 / max(0.5, float(args.sample_fps))
    last_fire_event_at = -1e9
    processed = 0
    start_wall = time.monotonic()
    next_sample_wall = start_wall

    emit({
        "type": "metadata",
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
    })

    try:
        while True:
            now_wall = time.monotonic()
            if now_wall < next_sample_wall:
                time.sleep(min(0.02, next_sample_wall - now_wall))
                continue

            elapsed = now_wall - start_wall
            if duration > 0 and elapsed >= duration:
                break

            # Stay synchronized to real video time. If inference takes longer than
            # one sample interval, jump forward instead of processing a backlog.
            target_frame = max(0, int(elapsed * fps))
            current_pos = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            if abs(target_frame - current_pos) > max(2, int(fps * 0.20)):
                cap.set(cv.CAP_PROP_POS_FRAMES, target_frame)

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_pos = max(0, int(cap.get(cv.CAP_PROP_POS_FRAMES)) - 1)
            video_time = frame_pos / fps
            processed += 1

            _seq, boxes, scores, classes = detector.process_frame_sync(frame_pos + 1, frame)
            h, w = frame.shape[:2]
            overlays = []
            fire_hits = []

            for box, score, cls_id in zip(boxes, scores, classes):
                cls_id = int(cls_id)
                label = fire.CLASSES[cls_id] if 0 <= cls_id < len(fire.CLASSES) else str(cls_id)
                if label not in ("fire", "smoke"):
                    continue
                x1, y1, x2, y2 = safe_box(box, w, h)
                overlays.append({
                    "label": label,
                    "score": float(score),
                    "box": [x1, y1, x2, y2],
                })
                if label == "fire":
                    fire_hits.append((box, float(score), cls_id))

            emit({
                "type": "detections",
                "video_time": video_time,
                "frame_width": w,
                "frame_height": h,
                "detections": overlays,
            })

            if duration > 0:
                emit({"type": "progress", "progress": min(99, int(video_time * 100.0 / duration))})

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

            # Keep sample cadence tied to wall clock. If inference was slow,
            # schedule from 'now' so we never build a queue of stale frames.
            next_sample_wall += interval
            finished = time.monotonic()
            if finished > next_sample_wall + interval:
                next_sample_wall = finished
    finally:
        cap.release()
        detector.stop()

    emit({"type": "detections", "video_time": duration, "frame_width": 0, "frame_height": 0, "detections": []})
    emit({"type": "progress", "progress": 100})
    emit({"type": "done", "processed_frames": processed})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FIRE DEMO ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
