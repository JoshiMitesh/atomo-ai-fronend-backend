#!/usr/bin/env python3
"""
LOW-LATENCY REAL-TIME MULTI-CAMERA FIRE / SMOKE DETECTOR
========================================================

Designed for Electron / VIM3 + KSNN/ASNN.

Goals:
- One NPU, strict round-robin inference.
- Continuous RTSP capture with latest-frame-only storage.
- Do not process stale frames.
- If the NPU is busy, cameras continue capturing.
- Skip intermediate frames rather than building latency.
- Optional GUI; --no-show keeps it headless.
- CPU-limited OpenCV/NumPy thread settings.
- Detection result is kept on the latest captured frame.
- The detector always tries to infer the newest frame available.

Important:
A single NPU cannot infer all cameras simultaneously. With a slow model,
per-camera detection latency is necessarily roughly:
    inference_time × number_of_cameras
This version minimizes *additional* capture/queue latency.

Model/output logic is kept compatible with the supplied fire code:
3 classes, classes-first, sigmoid, DFL 16 bins, class-aware NMS,
swapRB=False.
"""

# ============================================================================
# ENVIRONMENT
# ============================================================================

import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"

os.environ["OPENCV_FFMPEG_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

if "--no-show" not in sys.argv:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import math
import threading
import time
import traceback

import cv2 as cv
import numpy as np
import psutil

cv.setUseOptimized(True)
cv.setNumThreads(1)

try:
    cv.ocl.setUseOpenCL(False)
except Exception:
    pass

# ============================================================================
# KSNN / ASNN
# ============================================================================

try:
    from ksnn.api import KSNN as NPU_API
    from ksnn.types import output_format
    NPU_KIND = "KSNN"
except ImportError:
    try:
        from asnn.api import asnn as NPU_API
        from asnn.types import output_format
        NPU_KIND = "ASNN"
    except ImportError:
        sys.exit("[ERROR] Neither KSNN nor ASNN module found.")

# ============================================================================
# MODEL CONFIG
# ============================================================================

NUM_CLS = 3
LISTSIZE = NUM_CLS + 64

CLASSES = ("fire", "other", "smoke")

CLASS_COLORS = {
    "fire": (0, 0, 255),
    "smoke": (0, 165, 255),
    "other": (0, 255, 0),
}

HEAD_ORDER = "classes-first"
SCORE_ACTIVATION = "sigmoid"

OBJ_THRESH = 0.15
NMS_THRESH = 0.45
MAX_DETECTIONS = 100

DFL_BINS = np.arange(16, dtype=np.float32)

# ============================================================================
# LETTERBOX
# ============================================================================

def letterbox(frame, target=1024, canvas=None):
    h, w = frame.shape[:2]

    scale = target / max(h, w)

    nh = int(h * scale)
    nw = int(w * scale)

    resized = cv.resize(
        frame,
        (nw, nh),
        interpolation=cv.INTER_LINEAR,
    )

    if canvas is None:
        canvas = np.full(
            (target, target, 3),
            114,
            dtype=np.uint8,
        )
    else:
        canvas.fill(114)

    y0 = (target - nh) // 2
    x0 = (target - nw) // 2

    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    return canvas, scale, x0, y0

# ============================================================================
# UNLETTERBOX
# ============================================================================

def unletterbox_boxes(
    boxes,
    scale,
    x0,
    y0,
    orig_w,
    orig_h,
    target,
):
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = boxes.copy()

    boxes[:, 0] *= target
    boxes[:, 1] *= target
    boxes[:, 2] *= target
    boxes[:, 3] *= target

    boxes[:, 0] -= x0
    boxes[:, 2] -= x0
    boxes[:, 1] -= y0
    boxes[:, 3] -= y0

    boxes /= scale

    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

    return boxes

# ============================================================================
# ACTIVATIONS
# ============================================================================

def sigmoid(x):
    return 1.0 / (
        1.0 +
        np.exp(-np.clip(x, -80.0, 80.0))
    )

def softmax(x, axis=0):
    x = np.exp(
        x -
        np.max(
            x,
            axis=axis,
            keepdims=True,
        )
    )
    return x / np.maximum(
        x.sum(axis=axis, keepdims=True),
        1e-12,
    )

# ============================================================================
# YOLO FIRE/SMOKE POSTPROCESS
# ============================================================================

def process(input_data, target_res=1024):
    # FIXED: no "_, grid_h, grid_w" because output shape may be 2-D.
    if input_data.ndim != 3:
        raise RuntimeError(
            f"Unexpected output shape: {input_data.shape}"
        )

    _, grid_h, grid_w = input_data.shape

    stride = target_res / grid_w

    if HEAD_ORDER == "classes-first":
        class_logits = input_data[:NUM_CLS, :, :]
        dfl_source = input_data[
            NUM_CLS:NUM_CLS + 64,
            :,
            :,
        ]
    else:
        dfl_source = input_data[:64, :, :]
        class_logits = input_data[
            64:64 + NUM_CLS,
            :,
            :,
        ]

    if SCORE_ACTIVATION == "sigmoid":
        scores = sigmoid(class_logits)
    else:
        scores = np.clip(class_logits, 0.0, 1.0)

    max_scores = np.max(scores, axis=0)

    pos = np.where(max_scores >= OBJ_THRESH)

    if len(pos[0]) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    y_indices = pos[0]
    x_indices = pos[1]

    cand_scores = scores[:, y_indices, x_indices]

    cand_classes = np.argmax(cand_scores, axis=0)
    cand_max_scores = np.max(cand_scores, axis=0)

    dfl_data = dfl_source[:, y_indices, x_indices].T

    box_0 = softmax(dfl_data[:, 0:16], -1)
    box_1 = softmax(dfl_data[:, 16:32], -1)
    box_2 = softmax(dfl_data[:, 32:48], -1)
    box_3 = softmax(dfl_data[:, 48:64], -1)

    dfl_certainty = (
        box_0.max(axis=1) +
        box_1.max(axis=1) +
        box_2.max(axis=1) +
        box_3.max(axis=1)
    ) * 0.25

    dfl_left = np.dot(box_0, DFL_BINS)
    dfl_top = np.dot(box_1, DFL_BINS)
    dfl_right = np.dot(box_2, DFL_BINS)
    dfl_bottom = np.dot(box_3, DFL_BINS)

    x1 = (
        x_indices + 0.5 - dfl_left
    ) / grid_w

    y1 = (
        y_indices + 0.5 - dfl_top
    ) / grid_h

    x2 = (
        x_indices + 0.5 + dfl_right
    ) / grid_w

    y2 = (
        y_indices + 0.5 + dfl_bottom
    ) / grid_h

    min_size_norm = (
        stride * 0.1
    ) / target_res

    bw = x2 - x1
    bh = y2 - y1

    ratio = bw / (bh + 1e-6)

    valid = (
        (dfl_certainty >= 0.05) &
        (bw >= min_size_norm) &
        (bh >= min_size_norm) &
        (ratio >= 0.10) &
        (ratio <= 6.0) &
        (cand_max_scores >= OBJ_THRESH)
    )

    boxes = np.stack(
        [
            x1[valid],
            y1[valid],
            x2[valid],
            y2[valid],
        ],
        axis=-1,
    )

    return (
        boxes,
        cand_classes[valid],
        cand_max_scores[valid],
    )

# ============================================================================
# NMS
# ============================================================================

def _nms_single_class(boxes, scores):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (
        np.maximum(0.0, x2 - x1) *
        np.maximum(0.0, y2 - y1)
    )

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h

        union = (
            areas[i] +
            areas[order[1:]] -
            inter
        )

        ovr = inter / np.maximum(union, 1e-12)

        valid = np.where(ovr <= NMS_THRESH)[0]

        order = order[valid + 1]

    return np.asarray(keep, dtype=np.int32)

def nms_boxes(boxes, scores, classes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)

    keep = []

    for class_id in np.unique(classes):
        class_indices = np.flatnonzero(classes == class_id)

        local_keep = _nms_single_class(
            boxes[class_indices],
            scores[class_indices],
        )

        keep.extend(class_indices[local_keep])

    keep = np.asarray(keep, dtype=np.int32)

    if keep.size == 0:
        return keep

    keep = keep[
        np.argsort(scores[keep])[::-1]
    ]

    return keep[:MAX_DETECTIONS]

# ============================================================================
# COUNTS
# ============================================================================

def get_counts(classes):
    counts = {name: 0 for name in CLASSES}

    for class_id in classes:
        class_id = int(class_id)

        if 0 <= class_id < len(CLASSES):
            counts[CLASSES[class_id]] += 1

    return counts

# ============================================================================
# FIRE DETECTOR
# ============================================================================

class FireDetector:

    def __init__(self, model, library):
        self.model = os.path.abspath(model)
        self.library = os.path.abspath(library)

        self.target_res = 1024 if "1024" in self.model else 640

        self.yolov8 = None

        self.inference_count = 0

        self.last_npu_ms = 0.0
        self.last_pre_ms = 0.0
        self.last_post_ms = 0.0
        self.last_total_ms = 0.0

        self._lb_canvas = np.full(
            (
                self.target_res,
                self.target_res,
                3,
            ),
            114,
            dtype=np.uint8,
        )

    @staticmethod
    def _prepare_input(image):
        return cv.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            swapRB=False,
            crop=False,
        )[0]

    def initialize(self):
        if not os.path.isfile(self.model):
            raise FileNotFoundError(self.model)

        if not os.path.isfile(self.library):
            raise FileNotFoundError(self.library)

        if NPU_KIND == "KSNN":
            self.yolov8 = NPU_API("VIM3")
        else:
            self.yolov8 = NPU_API("Electron")

        print(f"[{NPU_KIND}] Initializing NPU...", flush=True)

        started = time.perf_counter()

        result = self.yolov8.nn_init(
            library=self.library,
            model=self.model,
            level=0,
        )

        elapsed = (
            time.perf_counter() -
            started
        ) * 1000.0

        print(
            f"[{NPU_KIND}] nn_init result: {result}",
            flush=True,
        )

        print(
            f"[{NPU_KIND}] initialization: {elapsed:.1f}ms",
            flush=True,
        )

        print(
            f"[{NPU_KIND}] model resolution: "
            f"{self.target_res}x{self.target_res}",
            flush=True,
        )

        print(
            f"[{NPU_KIND}] LISTSIZE: {LISTSIZE}",
            flush=True,
        )

    @staticmethod
    def _reshape_outputs(data):
        if data is None or len(data) != 3:
            raise RuntimeError(
                "Expected 3 NPU outputs, received "
                f"{0 if data is None else len(data)}"
            )

        sorted_tensors = sorted(
            data,
            key=lambda x: x.size,
            reverse=True,
        )

        reshaped = []

        for index, tensor in enumerate(sorted_tensors):
            if tensor.size % LISTSIZE != 0:
                raise RuntimeError(
                    f"Output {index}: {tensor.size} values "
                    f"not divisible by {LISTSIZE}"
                )

            cells = tensor.size // LISTSIZE

            grid_size = int(
                round(math.sqrt(cells))
            )

            if grid_size * grid_size != cells:
                raise RuntimeError(
                    f"Output {index}: invalid grid cells={cells}"
                )

            reshaped.append(
                tensor.reshape(
                    LISTSIZE,
                    grid_size,
                    grid_size,
                )
            )

        return reshaped

    def detect(self, frame):
        total_started = time.perf_counter()
        self.inference_count += 1

        orig_h, orig_w = frame.shape[:2]

        # PREPROCESS
        pre_started = time.perf_counter()

        (
            lb_img,
            scale,
            pad_x,
            pad_y,
        ) = letterbox(
            frame,
            target=self.target_res,
            canvas=self._lb_canvas,
        )

        input_data = self._prepare_input(lb_img)

        self.last_pre_ms = (
            time.perf_counter() -
            pre_started
        ) * 1000.0

        # NPU
        npu_started = time.perf_counter()

        data = self.yolov8.nn_inference(
            [input_data],
            platform="ONNX",
            reorder="2 1 0",
            output_tensor=3,
            output_format=output_format.OUT_FORMAT_FLOAT32,
        )

        self.last_npu_ms = (
            time.perf_counter() -
            npu_started
        ) * 1000.0

        # POSTPROCESS
        post_started = time.perf_counter()

        input_maps = self._reshape_outputs(data)

        all_boxes = []
        all_classes = []
        all_scores = []

        for grid_input in input_maps:
            (
                boxes,
                classes,
                scores,
            ) = process(
                grid_input,
                target_res=self.target_res,
            )

            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_classes.append(classes)
                all_scores.append(scores)

        if not all_boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int32)
        else:
            boxes = np.concatenate(all_boxes)
            classes = np.concatenate(all_classes)
            scores = np.concatenate(all_scores)

            keep = nms_boxes(
                boxes,
                scores,
                classes,
            )

            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

            boxes = unletterbox_boxes(
                boxes,
                scale,
                pad_x,
                pad_y,
                orig_w,
                orig_h,
                target=self.target_res,
            )

        self.last_post_ms = (
            time.perf_counter() -
            post_started
        ) * 1000.0

        self.last_total_ms = (
            time.perf_counter() -
            total_started
        ) * 1000.0

        return boxes, scores, classes

# ============================================================================
# LIVE RTSP SOURCE
# ============================================================================

class LiveSource:

    def __init__(self, name, source, hw_decode=True):
        self.name = name
        self.source = str(source)
        self.hw_decode = bool(hw_decode)

        self.cap = None
        self.running = True
        self.connected = False

        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_frame_id = 0

        self.latest_capture_time = 0.0

        self.capture_fps = 0.0

        self.frames_captured = 0
        self.frames_dropped_for_detection = 0
        self.reconnect_count = 0

        self.latest_boxes = np.zeros(
            (0, 4),
            dtype=np.float32,
        )
        self.latest_scores = np.zeros(
            (0,),
            dtype=np.float32,
        )
        self.latest_classes = np.zeros(
            (0,),
            dtype=np.int32,
        )

        self.last_detection_frame = 0
        self.last_detection_capture_frame = 0
        self.last_detection_age_ms = 0.0

        self.fire_count = 0
        self.other_count = 0
        self.smoke_count = 0

        self._reported_backend = False

        self._open()

        self.thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{name}",
            daemon=True,
        )

        self.thread.start()

    def _open_ffmpeg(self, src):
        params = []

        thread_property = getattr(
            cv,
            "CAP_PROP_N_THREADS",
            None,
        )

        if thread_property is not None:
            params.extend([
                thread_property,
                1,
            ])

        hw_property = getattr(
            cv,
            "CAP_PROP_HW_ACCELERATION",
            None,
        )

        hw_any = getattr(
            cv,
            "VIDEO_ACCELERATION_ANY",
            None,
        )

        if (
            self.hw_decode and
            hw_property is not None and
            hw_any is not None
        ):
            params.extend([
                hw_property,
                hw_any,
            ])

        try:
            if params:
                capture = cv.VideoCapture(
                    src,
                    cv.CAP_FFMPEG,
                    params,
                )
            else:
                capture = cv.VideoCapture(
                    src,
                    cv.CAP_FFMPEG,
                )
        except Exception:
            capture = cv.VideoCapture(
                src,
                cv.CAP_FFMPEG,
            )

        if not capture.isOpened() and params:
            capture.release()
            capture = cv.VideoCapture(
                src,
                cv.CAP_FFMPEG,
            )

        return capture

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        # Use environment options before opening.
        # UDP is lower latency on a clean LAN; TCP can be selected with
        # --rtsp-tcp when packet loss makes UDP unstable.
        transport = getattr(
            self,
            "_rtsp_transport",
            "udp",
        )

        os.environ[
            "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        ] = (
            f"rtsp_transport;{transport}|"
            "threads;1|"
            "fflags;nobuffer|"
            "flags;low_delay|"
            "max_delay;0|"
            "reorder_queue_size;0|"
            "buffer_size;32768"
        )

        self.cap = self._open_ffmpeg(self.source)

        if not self.cap.isOpened():
            self.cap = cv.VideoCapture(
                self.source,
                cv.CAP_FFMPEG,
            )

        # Best effort. Some OpenCV/FFmpeg backends ignore this property.
        try:
            self.cap.set(
                cv.CAP_PROP_BUFFERSIZE,
                1,
            )
        except Exception:
            pass

        if self.cap.isOpened():
            self.connected = True

            if not self._reported_backend:
                try:
                    backend = self.cap.getBackendName()
                except Exception:
                    backend = "unknown"

                hw_value = "unsupported"

                hw_property = getattr(
                    cv,
                    "CAP_PROP_HW_ACCELERATION",
                    None,
                )

                if hw_property is not None:
                    try:
                        hw_value = int(
                            self.cap.get(
                                hw_property
                            )
                        )
                    except Exception:
                        hw_value = "unknown"

                print(
                    f"[{self.name}] connected | "
                    f"backend={backend} | "
                    f"HW-request={self.hw_decode} | "
                    f"HW={hw_value} | "
                    f"decoder_threads=1",
                    flush=True,
                )

                self._reported_backend = True

        else:
            self.connected = False
            print(
                f"[{self.name}] connection failed",
                flush=True,
            )

    def _capture_loop(self):
        fps_count = 0
        fps_started = time.monotonic()

        while self.running:
            if (
                self.cap is None or
                not self.cap.isOpened()
            ):
                self.connected = False
                time.sleep(0.25)

                if self.running:
                    self.reconnect_count += 1
                    self._open()

                continue

            # grab/retrieve is intentional:
            # capture thread keeps draining the decoder even while NPU
            # inference is busy.
            grabbed = self.cap.grab()

            if not grabbed:
                self.connected = False
                time.sleep(0.05)

                if self.running:
                    self.reconnect_count += 1
                    self._open()

                continue

            ret, frame = self.cap.retrieve()

            if not ret or frame is None:
                continue

            now = time.monotonic()

            with self.lock:
                # The old frame is discarded immediately.
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.latest_capture_time = now
                self.frames_captured += 1

            fps_count += 1

            elapsed = now - fps_started

            if elapsed >= 2.0:
                self.capture_fps = fps_count / elapsed
                fps_count = 0
                fps_started = now

    def get_latest_frame(self, copy=False):
        with self.lock:
            frame = self.latest_frame
            frame_id = self.latest_frame_id
            capture_time = self.latest_capture_time

            if frame is None:
                return False, None, frame_id, capture_time

            if copy:
                frame = frame.copy()

            return True, frame, frame_id, capture_time

    def set_detection(
        self,
        boxes,
        scores,
        classes,
        scheduler_frame,
        capture_frame,
        capture_time,
    ):
        counts = get_counts(classes)

        age_ms = max(
            0.0,
            (
                time.monotonic() -
                capture_time
            ) * 1000.0,
        )

        with self.lock:
            self.latest_boxes = boxes.copy()
            self.latest_scores = scores.copy()
            self.latest_classes = classes.copy()

            self.last_detection_frame = scheduler_frame
            self.last_detection_capture_frame = capture_frame
            self.last_detection_age_ms = age_ms

            self.fire_count = counts.get("fire", 0)
            self.other_count = counts.get("other", 0)
            self.smoke_count = counts.get("smoke", 0)

    def get_detection(self):
        with self.lock:
            return (
                self.latest_boxes.copy(),
                self.latest_scores.copy(),
                self.latest_classes.copy(),
                self.last_detection_frame,
                self.last_detection_capture_frame,
                self.last_detection_age_ms,
                self.fire_count,
                self.other_count,
                self.smoke_count,
            )

    def stop(self):
        self.running = False

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass

# ============================================================================
# ROUND ROBIN WORKER
# ============================================================================

class RoundRobinNPUWorker:

    def __init__(
        self,
        sources,
        detector,
        stale_limit_ms=250.0,
    ):
        self.sources = sources
        self.detector = detector

        self.stale_limit_ms = float(
            stale_limit_ms
        )

        self.running = True
        self.camera_index = 0
        self.scheduler_frame = 0

        self.total_npu_fps = 0.0
        self.skipped_same_frame = 0
        self.skipped_stale = 0

        self.last_capture_ids = {
            id(source): -1
            for source in sources
        }

        self.thread = threading.Thread(
            target=self._loop,
            name="fire-npu-round-robin",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def _next_source(self):
        if not self.sources:
            return None

        # Strict round robin, but skip sources that have no new frame.
        for _ in range(len(self.sources)):
            source = self.sources[self.camera_index]

            self.camera_index = (
                self.camera_index + 1
            ) % len(self.sources)

            ok, frame, frame_id, capture_time = (
                source.get_latest_frame(copy=False)
            )

            if not ok or frame is None:
                continue

            previous = self.last_capture_ids[id(source)]

            if frame_id == previous:
                self.skipped_same_frame += 1
                continue

            return source, frame, frame_id, capture_time

        return None

    def _loop(self):
        fps_count = 0
        fps_started = time.monotonic()

        while self.running:
            selected = self._next_source()

            if selected is None:
                # NPU is never allowed to spin at 100% CPU just looking
                # for a new frame.
                time.sleep(0.002)
                continue

            source, frame, capture_frame, capture_time = selected

            self.last_capture_ids[
                id(source)
            ] = capture_frame

            # Check age before expensive inference.
            age_ms = max(
                0.0,
                (
                    time.monotonic() -
                    capture_time
                ) * 1000.0,
            )

            if (
                self.stale_limit_ms > 0 and
                age_ms > self.stale_limit_ms
            ):
                self.skipped_stale += 1
                continue

            self.scheduler_frame += 1
            seq = self.scheduler_frame

            try:
                (
                    boxes,
                    scores,
                    classes,
                ) = self.detector.detect(frame)

                source.set_detection(
                    boxes,
                    scores,
                    classes,
                    seq,
                    capture_frame,
                    capture_time,
                )

            except Exception as exc:
                print(
                    f"[NPU ERROR] {source.name} | "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()

                time.sleep(0.005)
                continue

            counts = get_counts(classes)

            detection_items = []

            for class_id, score in zip(
                classes[:10],
                scores[:10],
            ):
                cid = int(class_id)

                if 0 <= cid < len(CLASSES):
                    detection_items.append(
                        f"{CLASSES[cid]}:{float(score):.3f}"
                    )

            detection_text = (
                ", ".join(detection_items)
                if detection_items
                else "none"
            )

            actual_age_ms = max(
                0.0,
                (
                    time.monotonic() -
                    capture_time
                ) * 1000.0,
            )

            print(
                f"[DETECT] FRAME={seq} | "
                f"{source.name} | "
                f"capture_frame={capture_frame} | "
                f"age={actual_age_ms:.0f}ms | "
                f"fire={counts['fire']} | "
                f"other={counts['other']} | "
                f"smoke={counts['smoke']} | "
                f"total={len(boxes)} | "
                f"detections=[{detection_text}] | "
                f"pre={self.detector.last_pre_ms:.1f}ms | "
                f"NPU={self.detector.last_npu_ms:.1f}ms | "
                f"post={self.detector.last_post_ms:.1f}ms | "
                f"total={self.detector.last_total_ms:.1f}ms",
                flush=True,
            )

            fps_count += 1
            now = time.monotonic()
            elapsed = now - fps_started

            if elapsed >= 2.0:
                self.total_npu_fps = fps_count / elapsed
                fps_count = 0
                fps_started = now

    def stop(self):
        self.running = False

        try:
            self.thread.join(timeout=3.0)
        except Exception:
            pass

# ============================================================================
# DRAW
# ============================================================================

def draw_detections_fast(
    image,
    boxes,
    scores,
    classes,
):
    counts = {name: 0 for name in CLASSES}

    for box, score, class_id in zip(
        boxes,
        scores,
        classes,
    ):
        x1, y1, x2, y2 = box

        left = max(0, int(x1))
        top = max(0, int(y1))
        right = min(image.shape[1], int(x2))
        bottom = min(image.shape[0], int(y2))

        if right <= left or bottom <= top:
            continue

        class_id = int(class_id)

        cls_name = (
            CLASSES[class_id]
            if 0 <= class_id < len(CLASSES)
            else f"cls_{class_id}"
        )

        if cls_name in counts:
            counts[cls_name] += 1

        color = CLASS_COLORS.get(
            cls_name,
            (255, 0, 255),
        )

        cv.rectangle(
            image,
            (left, top),
            (right, bottom),
            color,
            2,
        )

        label = (
            f"{cls_name} "
            f"{float(score):.2f}"
        )

        (tw, th), _text_baseline = cv.getTextSize(
            label,
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )

        label_y = max(top - 5, th + 8)

        cv.rectangle(
            image,
            (
                left,
                label_y - th - 4,
            ),
            (
                left + tw + 5,
                label_y + 3,
            ),
            color,
            -1,
        )

        cv.putText(
            image,
            label,
            (left + 2, label_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    return counts

# ============================================================================
# GRID
# ============================================================================

def build_grid(frames, width, height):
    count = len(frames)

    if count == 0:
        return np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

    if count == 1:
        return cv.resize(
            frames[0],
            (width, height),
            interpolation=cv.INTER_NEAREST,
        )

    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))

    cell_w = max(1, width // cols)
    cell_h = max(1, height // rows)

    grid = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    for index, frame in enumerate(frames):
        row = index // cols
        col = index % cols

        resized = cv.resize(
            frame,
            (cell_w, cell_h),
            interpolation=cv.INTER_NEAREST,
        )

        y1 = row * cell_h
        x1 = col * cell_w
        y2 = min(y1 + cell_h, height)
        x2 = min(x1 + cell_w, width)

        grid[y1:y2, x1:x2] = resized[
            :y2-y1,
            :x2-x1,
        ]

    return grid

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Low-latency multi-camera fire/smoke detector"
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
    )

    parser.add_argument(
        "--software-decode",
        action="store_true",
    )

    parser.add_argument(
        "--rtsp-tcp",
        action="store_true",
        help="Use TCP instead of UDP for RTSP.",
    )

    parser.add_argument(
        "--stale-limit-ms",
        type=float,
        default=250.0,
        help=(
            "Skip a frame if it is older than this before "
            "inference. Use 0 to disable."
        ),
    )

    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--display-height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--display-fps",
        type=float,
        default=20.0,
    )

    args = parser.parse_args()

    global OBJ_THRESH
    global NMS_THRESH
    global MAX_DETECTIONS

    OBJ_THRESH = args.threshold
    NMS_THRESH = args.nms_threshold
    MAX_DETECTIONS = max(
        1,
        int(args.max_detections),
    )

    show_gui = not args.no_show

    print()
    print("=" * 62)
    print(" LOW-LATENCY MULTI-CAMERA FIRE/SMOKE DETECTOR")
    print("=" * 62)
    print(f"NPU              : {NPU_KIND}")
    print(f"Total cameras    : {len(args.inputs)}")
    print(f"Classes          : {CLASSES}")
    print(f"LISTSIZE         : {LISTSIZE}")
    print(f"Threshold        : {OBJ_THRESH}")
    print(f"NMS              : {NMS_THRESH}")
    print(f"Max detections   : {MAX_DETECTIONS}")
    print(f"GUI              : {'ON' if show_gui else 'OFF'}")
    print(
        "Hardware decode  : "
        f"{'OFF' if args.software_decode else 'REQUESTED'}"
    )
    print(
        f"RTSP transport   : "
        f"{'TCP' if args.rtsp_tcp else 'UDP'}"
    )
    print("Capture           : CONTINUOUS")
    print("Capture storage   : LATEST FRAME ONLY")
    print("NPU scheduler     : STRICT ROUND ROBIN")
    print(
        f"Stale limit       : "
        f"{args.stale_limit_ms:.0f}ms"
    )
    print("CPU threads       : 1")
    print("=" * 62)
    print()

    detector = FireDetector(
        args.model,
        args.library,
    )

    detector.initialize()

    sources = []

    for index, url in enumerate(args.inputs):
        name = f"Cam-{index + 1}"

        print(
            f"[SOURCE] {name}: {url}",
            flush=True,
        )

        source = LiveSource(
            name=name,
            source=url,
            hw_decode=not args.software_decode,
        )

        source._rtsp_transport = (
            "tcp" if args.rtsp_tcp else "udp"
        )

        # If _open() already happened, transport above is too late.
        # Re-open once with the requested transport.
        if source.cap is not None:
            try:
                source.cap.release()
            except Exception:
                pass

        source._reported_backend = False
        source._open()

        sources.append(source)

    worker = RoundRobinNPUWorker(
        sources=sources,
        detector=detector,
        stale_limit_ms=args.stale_limit_ms,
    )

    worker.start()

    window_name = (
        "Low-Latency Multi-Camera Fire & Smoke"
    )

    if show_gui:
        if not (
            os.environ.get("DISPLAY") or
            os.environ.get("WAYLAND_DISPLAY")
        ):
            print(
                "[GUI] No display server. Headless mode.",
                flush=True,
            )
            show_gui = False
        else:
            try:
                cv.namedWindow(
                    window_name,
                    cv.WINDOW_NORMAL,
                )
                cv.resizeWindow(
                    window_name,
                    args.display_width,
                    args.display_height,
                )
                print(
                    "[GUI] Press q to quit.",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[GUI ERROR] {exc}",
                    flush=True,
                )
                show_gui = False

    process_proc = psutil.Process(os.getpid())
    process_proc.cpu_percent(interval=None)

    metrics_time = time.monotonic()

    display_interval = (
        1.0 /
        max(1.0, args.display_fps)
    )

    next_display = time.monotonic()

    try:
        while True:
            now = time.monotonic()

            if show_gui and now >= next_display:
                next_display = now + display_interval

                display_frames = []

                total_fire = 0
                total_other = 0
                total_smoke = 0

                for source in sources:
                    (
                        ok,
                        frame,
                        capture_frame,
                        capture_time,
                    ) = source.get_latest_frame(copy=True)

                    (
                        boxes,
                        scores,
                        classes,
                        detection_frame,
                        detection_capture_frame,
                        detection_age_ms,
                        fire_count,
                        other_count,
                        smoke_count,
                    ) = source.get_detection()

                    total_fire += fire_count
                    total_other += other_count
                    total_smoke += smoke_count

                    if not ok or frame is None:
                        frame = np.zeros(
                            (360, 640, 3),
                            dtype=np.uint8,
                        )

                        cv.putText(
                            frame,
                            f"{source.name}: NO SIGNAL",
                            (20, 60),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv.LINE_AA,
                        )
                    else:
                        draw_detections_fast(
                            frame,
                            boxes,
                            scores,
                            classes,
                        )

                    hud = (
                        f"{source.name} | "
                        f"F:{fire_count} "
                        f"O:{other_count} "
                        f"S:{smoke_count} | "
                        f"Det:{detection_frame} | "
                        f"Age:{detection_age_ms:.0f}ms"
                    )

                    cv.rectangle(
                        frame,
                        (5, 5),
                        (
                            min(
                                frame.shape[1] - 5,
                                700,
                            ),
                            40,
                        ),
                        (0, 0, 0),
                        -1,
                    )

                    cv.putText(
                        frame,
                        hud,
                        (12, 29),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv.LINE_AA,
                    )

                    display_frames.append(frame)

                mosaic = build_grid(
                    display_frames,
                    args.display_width,
                    args.display_height,
                )

                global_text = (
                    f"NPU FPS:{worker.total_npu_fps:.2f} | "
                    f"Fire:{total_fire} | "
                    f"Other:{total_other} | "
                    f"Smoke:{total_smoke}"
                )

                cv.rectangle(
                    mosaic,
                    (0, 0),
                    (mosaic.shape[1], 34),
                    (0, 0, 0),
                    -1,
                )

                cv.putText(
                    mosaic,
                    global_text,
                    (10, 24),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv.LINE_AA,
                )

                cv.imshow(
                    window_name,
                    mosaic,
                )

                key = cv.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            else:
                time.sleep(0.002)

            now = time.monotonic()

            if now - metrics_time >= 5.0:
                cpu = process_proc.cpu_percent(
                    interval=None
                )

                total_capture_fps = sum(
                    source.capture_fps
                    for source in sources
                )

                total_fire = 0
                total_other = 0
                total_smoke = 0

                fire_parts = []
                other_parts = []
                smoke_parts = []

                age_parts = []

                for source in sources:
                    (
                        boxes,
                        scores,
                        classes,
                        detection_frame,
                        detection_capture_frame,
                        detection_age_ms,
                        fire_count,
                        other_count,
                        smoke_count,
                    ) = source.get_detection()

                    total_fire += fire_count
                    total_other += other_count
                    total_smoke += smoke_count

                    fire_parts.append(
                        f"{source.name}={fire_count}"
                    )
                    other_parts.append(
                        f"{source.name}={other_count}"
                    )
                    smoke_parts.append(
                        f"{source.name}={smoke_count}"
                    )

                    age_parts.append(
                        f"{source.name}={detection_age_ms:.0f}ms"
                    )

                print()
                print(
                    f"[METRICS] "
                    f"CPU={cpu:.1f}% | "
                    f"CAPTURE={total_capture_fps:.1f}fps | "
                    f"NPU={worker.total_npu_fps:.2f}fps | "
                    f"PRE={detector.last_pre_ms:.1f}ms | "
                    f"NPU_TIME={detector.last_npu_ms:.1f}ms | "
                    f"POST={detector.last_post_ms:.1f}ms | "
                    f"TOTAL={detector.last_total_ms:.1f}ms",
                    flush=True,
                )

                print(
                    "[AGE] " +
                    " | ".join(age_parts),
                    flush=True,
                )

                print(
                    "[FIRE COUNT] " +
                    " | ".join(fire_parts),
                    flush=True,
                )

                print(
                    "[OTHER COUNT] " +
                    " | ".join(other_parts),
                    flush=True,
                )

                print(
                    "[SMOKE COUNT] " +
                    " | ".join(smoke_parts),
                    flush=True,
                )

                print(
                    f"[TOTAL] Fire={total_fire} | "
                    f"Other={total_other} | "
                    f"Smoke={total_smoke}",
                    flush=True,
                )

                print(
                    f"[SCHEDULER] "
                    f"same-frame-skips={worker.skipped_same_frame} | "
                    f"stale-skips={worker.skipped_stale}",
                    flush=True,
                )

                metrics_time = now

    except KeyboardInterrupt:
        print(
            "\n[INFO] Ctrl+C received",
            flush=True,
        )

    finally:
        print(
            "[INFO] Stopping NPU worker...",
            flush=True,
        )

        worker.stop()

        print(
            "[INFO] Stopping streams...",
            flush=True,
        )

        for source in sources:
            source.stop()

        if show_gui:
            try:
                cv.destroyAllWindows()
            except Exception:
                pass

        print(
            "[INFO] Done.",
            flush=True,
        )

if __name__ == "__main__":
    main()
