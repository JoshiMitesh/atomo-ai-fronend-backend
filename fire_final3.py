#!/usr/bin/env python3
"""Low-CPU real-time three-class fire/smoke detector for the KSNN NPU.

The capture, inference, and display paths are asynchronous. Only the newest
raw frame is submitted to the NPU; stale frames are dropped deliberately.
"""

import os
import sys
import platform
import subprocess
import traceback

# Safe Qt environment settings
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

os.environ["OPENCV_FFMPEG_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import argparse
import threading
import time
from collections import deque
from queue import Queue, Empty, Full
import cv2 as cv

cv.setUseOptimized(True)
# One OpenCV worker keeps CPU usage predictable on the edge board. The NPU
# remains responsible for neural-network execution.
cv.setNumThreads(1)

try:
    from ksnn.api import KSNN as NPU_API
    from ksnn.types import output_format
    NPU_KIND = 'KSNN'
except ImportError:
    try:
        from asnn.api import asnn as NPU_API
        from asnn.types import output_format
        NPU_KIND = 'ASNN'
    except ImportError:
        sys.exit('[ERROR] Neither KSNN nor ASNN module found.')

# Model configuration. The order must match the class order used for training.
NUM_CLS  = 3
LISTSIZE = NUM_CLS + 64
CLASSES  = ("fire", "other", "smoke")

# OpenCV uses BGR colours.
CLASS_COLORS = {
    "fire":  (0, 0, 255),
    "smoke": (0, 165, 255),
    "other": (0, 255, 0),
}

# Generated models can expose either [classes, 64 DFL] or [64 DFL, classes].
HEAD_ORDER = "classes-first"
SCORE_ACTIVATION = "sigmoid"

OBJ_THRESH = 0.15
NMS_THRESH = 0.45
MAX_DETECTIONS = 100

DFL_BINS = np.arange(16, dtype=np.float32)


def _run_diagnostic_command(command):
    """Run a read-only system diagnostic and return printable output."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
        output = result.stdout.strip() or "<no output>"
        return result.returncode, output
    except FileNotFoundError:
        return 127, f"command not installed: {command[0]}"
    except Exception as exc:
        return 1, f"diagnostic command failed: {exc}"


def print_preflight_diagnostics(model, library):
    print("\n========== NPU PREFLIGHT DIAGNOSTICS ==========", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Machine: {platform.machine()} | Kernel: {platform.release()}", flush=True)
    print(f"NPU Python API: {NPU_KIND}", flush=True)
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '<not set>')}", flush=True)

    for label, path in (("NB model", model), ("SO library", library)):
        real_path = os.path.realpath(path)
        exists = os.path.isfile(real_path)
        readable = os.access(real_path, os.R_OK) if exists else False
        size = os.path.getsize(real_path) if exists else 0
        print(
            f"{label}: path={real_path} | exists={exists} | readable={readable} | bytes={size}",
            flush=True,
        )
        if not exists:
            raise FileNotFoundError(f"{label} does not exist: {real_path}")
        if size == 0:
            raise RuntimeError(f"{label} is empty: {real_path}")

    galcore = "/dev/galcore"
    print(
        f"NPU device: {galcore} | exists={os.path.exists(galcore)} | "
        f"readable={os.access(galcore, os.R_OK)} | writable={os.access(galcore, os.W_OK)}",
        flush=True,
    )

    for command in (("file", library), ("file", model), ("ldd", "-r", library)):
        returncode, output = _run_diagnostic_command(list(command))
        print(f"\n$ {' '.join(command)}\n[exit={returncode}]\n{output}", flush=True)

    print("========== END PREFLIGHT DIAGNOSTICS ==========\n", flush=True)

# ─── Letterbox ────────────────────────────────────────────────────────────────

def letterbox(frame, target=1024, canvas=None):
    h, w   = frame.shape[:2]
    scale  = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    # INTER_LINEAR is fast and preserves small fire/smoke regions well.
    resized = cv.resize(frame, (nw, nh), interpolation=cv.INTER_LINEAR)
    if canvas is None:
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    else:
        canvas[:] = 114
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas, scale, x0, y0

def unletterbox_boxes(boxes, scale, x0, y0, orig_w, orig_h, target):
    boxes = boxes.copy()
    boxes[:, 0] *= target
    boxes[:, 1] *= target
    boxes[:, 2] *= target
    boxes[:, 3] *= target
    boxes[:, 0] -= x0;  boxes[:, 2] -= x0
    boxes[:, 1] -= y0;  boxes[:, 3] -= y0
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
    return boxes

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -80.0, 80.0)))

def softmax(x, axis=0):
    x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x / x.sum(axis=axis, keepdims=True)

def process(input_data, target_res=1024):
    _, grid_h, grid_w = input_data.shape[:3]
    stride = target_res / grid_w

    if HEAD_ORDER == "classes-first":
        class_logits = input_data[:NUM_CLS, :, :]
        dfl_source = input_data[NUM_CLS:NUM_CLS+64, :, :]
    else:
        dfl_source = input_data[:64, :, :]
        class_logits = input_data[64:64+NUM_CLS, :, :]

    if SCORE_ACTIVATION == "sigmoid":
        scores = sigmoid(class_logits)
    else:
        scores = np.clip(class_logits, 0.0, 1.0)

    thresh = OBJ_THRESH
    max_scores = np.max(scores, axis=0)
    pos = np.where(max_scores >= thresh)

    if len(pos[0]) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    y_indices, x_indices = pos[0], pos[1]

    cand_scores = scores[:, y_indices, x_indices]
    cand_classes = np.argmax(cand_scores, axis=0)
    cand_max_scores = np.max(cand_scores, axis=0)

    dfl_data = dfl_source[:, y_indices, x_indices].T

    box_0 = softmax(dfl_data[:, 0:16], -1)
    box_1 = softmax(dfl_data[:, 16:32], -1)
    box_2 = softmax(dfl_data[:, 32:48], -1)
    box_3 = softmax(dfl_data[:, 48:64], -1)

    dfl_certainty = (box_0.max(axis=1) + box_1.max(axis=1) + box_2.max(axis=1) + box_3.max(axis=1)) * 0.25

    dfl_left   = np.dot(box_0, DFL_BINS)
    dfl_top    = np.dot(box_1, DFL_BINS)
    dfl_right  = np.dot(box_2, DFL_BINS)
    dfl_bottom = np.dot(box_3, DFL_BINS)

    x1 = (x_indices + 0.5 - dfl_left) / grid_w
    y1 = (y_indices + 0.5 - dfl_top) / grid_h
    x2 = (x_indices + 0.5 + dfl_right) / grid_w
    y2 = (y_indices + 0.5 + dfl_bottom) / grid_h

    min_size_norm = (stride * 0.1) / target_res
    bw = x2 - x1
    bh = y2 - y1
    # Fire and smoke can have irregular shapes, so keep a broad ratio range.
    valid = (dfl_certainty >= 0.05) & (bw >= min_size_norm) & (bh >= min_size_norm) & \
            (bw / (bh + 1e-6) >= 0.10) & (bw / (bh + 1e-6) <= 6.0)

    boxes = np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=-1)
    return boxes, cand_classes[valid], cand_max_scores[valid]

def _nms_single_class(boxes, scores):
    """Return local indices retained for one class."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w1   = np.maximum(0.0, xx2 - xx1)
        h1   = np.maximum(0.0, yy2 - yy1)
        inter = w1 * h1
        union = areas[i] + areas[order[1:]] - inter
        ovr   = inter / np.maximum(union, 1e-12)
        order = order[np.where(ovr <= NMS_THRESH)[0] + 1]
    return np.asarray(keep, dtype=np.int32)


def nms_boxes(boxes, scores, classes):
    """Class-aware NMS preserves valid overlapping fire and smoke boxes."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)

    keep = []
    for class_id in np.unique(classes):
        class_indices = np.flatnonzero(classes == class_id)
        local_keep = _nms_single_class(boxes[class_indices], scores[class_indices])
        keep.extend(class_indices[local_keep])

    keep = np.asarray(keep, dtype=np.int32)
    if keep.size == 0:
        return keep

    # Draw and return the strongest detections first, and bound worst-case CPU
    # if a bad frame produces an unusually large candidate set.
    keep = keep[np.argsort(scores[keep])[::-1]]
    return keep[:MAX_DETECTIONS]

# ─── Fast Drawing ─────────────────────────────────────────────────────────────

def draw_detections_fast(image, detections, scale_x=1.0, scale_y=1.0):
    class_counts = {name: 0 for name in CLASSES}
    if detections is not None:
        if len(detections) == 4:
            _, boxes_px, scores, classes = detections
        else:
            boxes_px, scores, classes = detections
        for box, score, cl in zip(boxes_px, scores, classes):
            x1, y1, x2, y2 = box
            left   = max(0, int(x1 * scale_x))
            top    = max(0, int(y1 * scale_y))
            right  = min(image.shape[1], int(x2 * scale_x))
            bottom = min(image.shape[0], int(y2 * scale_y))

            if right <= left or bottom <= top:
                continue

            class_id = int(cl)
            cls_name = CLASSES[class_id] if 0 <= class_id < len(CLASSES) else f"cls_{class_id}"
            if cls_name in class_counts:
                class_counts[cls_name] += 1

            color = CLASS_COLORS.get(cls_name.lower(), (255, 0, 255))
            cv.rectangle(image, (left, top), (right, bottom), color, 3)

            label = f"{cls_name.strip()} {score:.2f}"

            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            label_y = max(top - 5, text_height + 10)
            label_x = left

            cv.rectangle(image, (label_x, label_y - text_height - 5), (label_x + text_width + 5, label_y + 5), color, -1)
            cv.putText(image, label, (label_x + 1, label_y + 1), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 1)
            cv.putText(image, label, (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # HUD Box Top-Left
    hud_text = "  ".join(f"{name.title()}: {class_counts[name]}" for name in CLASSES)
    (tw, th), _ = cv.getTextSize(hud_text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv.rectangle(image, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 0), -1)
    cv.rectangle(image, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 255, 0), 2)
    cv.putText(image, hud_text, (20, 10 + th + 8), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def prepare_display_frame(frame, max_width):
    """Create a display-only frame without changing the inference image."""
    height, width = frame.shape[:2]
    if max_width > 0 and width > max_width:
        display_width = int(max_width)
        display_height = max(1, int(round(height * display_width / width)))
        display = cv.resize(
            frame,
            (display_width, display_height),
            interpolation=cv.INTER_AREA,
        )
        return display, display_width / width, display_height / height
    return frame.copy(), 1.0, 1.0

# ─── Low-CPU Rate-Governed RTSP Reader Thread ────────────────────────────────

class RTSPStreamReaderLowCPU:
    def __init__(self, source, read_fps=30, hw_decode=True):
        self.source = source
        self.read_fps = max(1.0, float(read_fps))
        self.hw_decode = bool(hw_decode)
        self.is_file = os.path.isfile(str(source))
        self.frame = None
        self.seq = 0
        self.eof = False
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.running = False
        self._reported_backend = False
        self._open()

    def _open_ffmpeg(self, src):
        """Request hardware decode and one decoder thread when supported."""
        params = []
        thread_property = getattr(cv, "CAP_PROP_N_THREADS", None)
        if thread_property is not None:
            params.extend([thread_property, 1])

        hw_property = getattr(cv, "CAP_PROP_HW_ACCELERATION", None)
        hw_any = getattr(cv, "VIDEO_ACCELERATION_ANY", None)
        if self.hw_decode and hw_property is not None and hw_any is not None:
            params.extend([hw_property, hw_any])

        try:
            if params:
                capture = cv.VideoCapture(src, cv.CAP_FFMPEG, params)
            else:
                capture = cv.VideoCapture(src, cv.CAP_FFMPEG)
        except Exception as exc:
            print(f"[VIDEO] Accelerated open unavailable: {exc}", flush=True)
            capture = cv.VideoCapture(src, cv.CAP_FFMPEG)

        # Some OpenCV/FFmpeg builds reject unsupported open-only parameters.
        # Retry without them rather than breaking an already working stream.
        if not capture.isOpened() and params:
            capture.release()
            capture = cv.VideoCapture(src, cv.CAP_FFMPEG)
        return capture

    def _open(self):
        src = str(self.source)
        old_capture = getattr(self, "cap", None)
        if old_capture is not None:
            old_capture.release()

        if src.startswith("rtsp://") or src.startswith("http://") or src.startswith("https://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|threads;1|fflags;nobuffer|flags;low_delay|max_delay;0|buffer_size;65536"
            self.cap = self._open_ffmpeg(src)
        else:
            self.cap = self._open_ffmpeg(src)

        if not self.cap.isOpened():
            self.cap = cv.VideoCapture(src)
        
        if not self.is_file:
            # Downscale capture resolution for live RTSP streams only
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            source_fps = float(self.cap.get(cv.CAP_PROP_FPS))
            if source_fps > 0.0:
                self.read_fps = source_fps
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened() and not self._reported_backend:
            try:
                backend = self.cap.getBackendName()
            except Exception:
                backend = "unknown"
            hw_value = "unsupported"
            hw_property = getattr(cv, "CAP_PROP_HW_ACCELERATION", None)
            if hw_property is not None:
                try:
                    hw_value = str(int(self.cap.get(hw_property)))
                except Exception:
                    hw_value = "unknown"
            print(
                f"[VIDEO] backend={backend} hardware_decode_requested={self.hw_decode} "
                f"hw_acceleration={hw_value} decoder_threads_requested=1",
                flush=True,
            )
            self._reported_backend = True

    def start(self):
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._reader,
            name="video-capture-reader",
            daemon=True,
        )
        self.thread.start()
        return self

    def _reader(self):
        fails = 0
        target_fps_interval = 1.0 / float(self.read_fps)
        
        while self.running:
            t0 = time.perf_counter()
            if self.cap is None or not self.cap.isOpened():
                self._open()

            # Fast grab() skips YUV->BGR conversion
            grabbed = self.cap.grab()
            if not grabbed:
                if self.is_file:
                    with self.frame_ready:
                        self.eof = True
                        self.running = False
                        self.frame_ready.notify_all()
                    break

                if not grabbed:
                    fails += 1
                    if fails % 10 == 0:
                        print(f"[Stream Reader] Reconnecting stream ({fails})...", flush=True)
                    if self.cap:
                        self.cap.release()
                    self.stop_event.wait(0.5)
                    continue

            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                fails = 0
                with self.frame_ready:
                    self.seq += 1
                    self.frame = frame
                    self.frame_ready.notify_all()

            elapsed = time.perf_counter() - t0
            sleep_time = target_fps_interval - elapsed
            if sleep_time > 0:
                self.stop_event.wait(sleep_time)

    def read(self, last_seq=None, timeout=0.0):
        """Wait for and return a new immutable raw frame reference.

        The capture thread replaces ``self.frame``; it never mutates an already
        published frame. Avoiding a copy here saves CPU. The GUI makes its own
        copy before drawing, so inference always receives clean pixels.
        """
        with self.frame_ready:
            if timeout > 0.0:
                self.frame_ready.wait_for(
                    lambda: (
                        (self.frame is not None and (last_seq is None or self.seq != last_seq))
                        or self.eof
                        or not self.running
                    ),
                    timeout=timeout,
                )
            if self.frame is None:
                return False, None, -1
            if last_seq is not None and self.seq == last_seq:
                return False, None, self.seq
            return True, self.frame, self.seq

    def stop(self):
        self.running = False
        self.stop_event.set()
        with self.frame_ready:
            self.frame_ready.notify_all()
        if hasattr(self, 'cap'):
            self.cap.release()

# ─── Async NPU Detection Worker ───────────────────────────────────────────────

class DetectionWorker:
    def __init__(self, model, library, level=0, process_interval=0.100,
                 debug=False, log_every=1, result_log_every=10):
        self.model = model
        self.library = library
        self.level = level
        self.debug = bool(debug)
        self.log_every = max(1, int(log_every))
        self.result_log_every = max(1, int(result_log_every))
        self.inference_count = 0
        self.process_interval = max(0.0, float(process_interval))
        self.yolov8 = None
        self.running           = False
        self.lock              = threading.Lock()
        self.latest_detections = None
        self.frame_queue = Queue(maxsize=1)
        self.completion_times = deque(maxlen=30)
        self.last_inference_ms = 0.0

        if "1024" in str(self.model):
            self.target_res = 1024
        else:
            self.target_res = 640

        self._lb_canvas = np.full((self.target_res, self.target_res, 3), 114, dtype=np.uint8)

    def _prepare_input(self, image):
        """Use OpenCV's compiled BGR-HWC to normalized BGR-CHW conversion."""
        # Return blob[0] directly. Avoiding a second copy is important for the
        # 3x1024x1024 float32 input while preserving the verified colour order.
        return cv.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            swapRB=False,
            crop=False,
        )[0]

    def initialize(self):
        if self.debug:
            print_preflight_diagnostics(self.model, self.library)
        if NPU_KIND == 'KSNN':
            self.yolov8 = NPU_API('VIM3')
        else:
            self.yolov8 = NPU_API('Electron')
        print(
            f'[{NPU_KIND}] Initializing NPU Pipeline: platform object={type(self.yolov8).__name__}, '
            f'level={self.level}...',
            flush=True,
        )
        started = time.perf_counter()
        try:
            init_result = self.yolov8.nn_init(
                library=self.library,
                model=self.model,
                level=self.level,
            )
        except Exception:
            print(f'[{NPU_KIND}] nn_init FAILED after {time.perf_counter() - started:.3f}s', flush=True)
            traceback.print_exc()
            raise
        print(
            f'[{NPU_KIND}] nn_init returned {init_result!r} in '
            f'{time.perf_counter() - started:.3f}s',
            flush=True,
        )
        print(
            f'[{NPU_KIND}] Loaded model at target={self.target_res}x{self.target_res}',
            flush=True,
        )

    def start(self):
        self.running = True
        if self.yolov8 is None:
            self.initialize()
        threading.Thread(target=self._detection_loop, daemon=True).start()
        return self

    @staticmethod
    def _reshape_outputs(data):
        if data is None or len(data) != 3:
            raise RuntimeError(f"Expected 3 NPU output tensors, received {0 if data is None else len(data)}")

        sorted_tensors = sorted(data, key=lambda d: d.size, reverse=True)
        reshaped = []
        for index, tensor in enumerate(sorted_tensors):
            if tensor.size % LISTSIZE != 0:
                raise RuntimeError(
                    f"Output {index} has {tensor.size} values, which is not divisible by "
                    f"{LISTSIZE} channels ({NUM_CLS} classes + 64 DFL). "
                    "Check the class count and use the matching .nb/.so pair."
                )

            cells = tensor.size // LISTSIZE
            grid_size = int(round(np.sqrt(cells)))
            if grid_size * grid_size != cells:
                raise RuntimeError(
                    f"Output {index} cannot form a square grid: values={tensor.size}, "
                    f"channels={LISTSIZE}, cells={cells}"
                )
            reshaped.append(tensor.reshape(LISTSIZE, grid_size, grid_size))
        return reshaped

    def _should_log(self, inference_number):
        return self.debug and (
            inference_number <= 3 or inference_number % self.log_every == 0
        )

    @staticmethod
    def _array_summary(array):
        array = np.asarray(array)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return (
                f"shape={array.shape} dtype={array.dtype} size={array.size} "
                "finite=0 min=NA max=NA mean=NA"
            )
        return (
            f"shape={array.shape} dtype={array.dtype} size={array.size} "
            f"finite={finite.size}/{array.size} min={float(finite.min()):.6f} "
            f"max={float(finite.max()):.6f} mean={float(finite.mean()):.6f}"
        )

    def _log_outputs(self, data, inference_number):
        print(
            f"[DEBUG #{inference_number}] NPU returned "
            f"{0 if data is None else len(data)} output tensors",
            flush=True,
        )
        if data is None:
            return
        for index, tensor in enumerate(data):
            print(
                f"[DEBUG #{inference_number}] raw_output[{index}]: "
                f"{self._array_summary(tensor)}",
                flush=True,
            )

    def _decode_outputs(self, data, target_res, inference_number, log_now):
        input_data = self._reshape_outputs(data)
        all_boxes, all_classes, all_scores = [], [], []

        for scale_index, grid_input in enumerate(input_data):
            first_class_candidate = grid_input[:NUM_CLS]
            last_class_candidate = grid_input[64:64+NUM_CLS]
            if HEAD_ORDER == "classes-first":
                raw_class = first_class_candidate
            else:
                raw_class = last_class_candidate

            boxes, classes, scores = process(grid_input, target_res=target_res)
            if log_now:
                activated = sigmoid(raw_class) if SCORE_ACTIVATION == "sigmoid" else np.clip(raw_class, 0.0, 1.0)
                top_score = float(activated.max()) if activated.size else float("nan")
                first_top = float(sigmoid(first_class_candidate).max())
                last_top = float(sigmoid(last_class_candidate).max())
                print(
                    f"[DEBUG #{inference_number}] scale={scale_index} "
                    f"grid={grid_input.shape[1]}x{grid_input.shape[2]} "
                    f"raw_class=({self._array_summary(raw_class)}) "
                    f"top_score={top_score:.6f} candidates={len(boxes)} "
                    f"diagnostic_sigmoid_top(first_channels)={first_top:.6f} "
                    f"diagnostic_sigmoid_top(last_channels)={last_top:.6f}",
                    flush=True,
                )
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_classes.append(classes)
                all_scores.append(scores)

        if not all_boxes:
            if log_now:
                print(
                    f"[DEBUG #{inference_number}] No raw candidate reached threshold "
                    f"{OBJ_THRESH:.4f}",
                    flush=True,
                )
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        boxes = np.concatenate(all_boxes)
        classes = np.concatenate(all_classes)
        scores = np.concatenate(all_scores)
        raw_count = len(boxes)
        keep = nms_boxes(boxes, scores, classes)

        if log_now:
            print(
                f"[DEBUG #{inference_number}] candidates_before_nms={raw_count} "
                f"detections_after_nms={len(keep)}",
                flush=True,
            )
        return boxes[keep], scores[keep], classes[keep]

    def _detection_loop(self):
        target_res = self.target_res
        frame_cnt = 0
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.20)
            except Empty:
                continue

            try:
                if frame_data is None:
                    continue

                seq, orig_img = frame_data
                if orig_img is None:
                    continue

                orig_h, orig_w = orig_img.shape[:2]
                frame_cnt += 1
                self.inference_count += 1
                inference_number = self.inference_count
                log_now = self._should_log(inference_number)
                inference_cycle_started = time.perf_counter()

                lb_img, scale, pad_x, pad_y = letterbox(orig_img, target=target_res, canvas=self._lb_canvas)
                input_data = self._prepare_input(lb_img)
                preprocess_ms = (time.perf_counter() - inference_cycle_started) * 1000.0

                if log_now:
                    print(
                        f"\n[DEBUG #{inference_number}] frame_seq={seq} "
                        f"original={orig_w}x{orig_h} letterbox_scale={scale:.6f} "
                        f"padding=({pad_x},{pad_y}) input=({self._array_summary(input_data)})",
                        flush=True,
                    )

                npu_started = time.perf_counter()
                data = self.yolov8.nn_inference(
                    [input_data],
                    platform='ONNX',
                    reorder='2 1 0',
                    output_tensor=3,
                    output_format=output_format.OUT_FORMAT_FLOAT32
                )
                npu_ms = (time.perf_counter() - npu_started) * 1000.0
                self.last_inference_ms = npu_ms

                if log_now:
                    print(f"[DEBUG #{inference_number}] nn_inference_time={npu_ms:.3f} ms", flush=True)
                    self._log_outputs(data, inference_number)

                postprocess_started = time.perf_counter()
                norm_boxes, curr_scores, curr_classes = self._decode_outputs(
                    data, target_res, inference_number, log_now
                )
                if len(norm_boxes) > 0:
                    curr_boxes = unletterbox_boxes(
                        norm_boxes, scale, pad_x, pad_y, orig_w, orig_h, target=target_res
                    )
                else:
                    curr_boxes = np.zeros((0, 4), dtype=np.float32)
                postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0

                num_dets = len(curr_boxes)
                with self.lock:
                    self.latest_detections = (seq, curr_boxes, curr_scores, curr_classes)

                completed_at = time.perf_counter()
                self.completion_times.append(completed_at)

                result_log_now = inference_number % self.result_log_every == 0
                if log_now or result_log_now:
                    counts = {name: 0 for name in CLASSES}
                    for class_id in curr_classes:
                        class_id = int(class_id)
                        if 0 <= class_id < len(CLASSES):
                            counts[CLASSES[class_id]] += 1
                    count_text = ", ".join(f"{name}={counts[name]}" for name in CLASSES)
                    measured_fps = (
                        (len(self.completion_times) - 1)
                        / (self.completion_times[-1] - self.completion_times[0])
                        if len(self.completion_times) >= 2
                        and self.completion_times[-1] > self.completion_times[0]
                        else 0.0
                    )
                    total_ms = (time.perf_counter() - inference_cycle_started) * 1000.0
                    detection_text = (
                        ", ".join(
                            f"{CLASSES[int(class_id)]}:{float(score):.3f}"
                            for class_id, score in zip(curr_classes[:10], curr_scores[:10])
                            if 0 <= int(class_id) < len(CLASSES)
                        )
                        if len(curr_scores) else "none"
                    )
                    print(
                        f"[DETECTION #{inference_number}] frame={seq} | {count_text}, "
                        f"total={num_dets} | detections=[{detection_text}] | "
                        f"pre={preprocess_ms:.1f}ms npu={npu_ms:.1f}ms "
                        f"post={postprocess_ms:.1f}ms total={total_ms:.1f}ms "
                        f"rate={measured_fps:.2f} FPS",
                        flush=True,
                    )

                if self.process_interval > 0:
                    # process_interval is a minimum start-to-start period, not
                    # an extra delay after inference. Slow NPU calls therefore
                    # receive no unnecessary sleep.
                    remaining = self.process_interval - (
                        time.perf_counter() - inference_cycle_started
                    )
                    if remaining > 0:
                        time.sleep(remaining)

            except Exception as e:
                if "Empty" not in str(e):
                    print(f'[Detection Error]: {type(e).__name__}: {e}', flush=True)
                    if self.debug:
                        traceback.print_exc()

    def process_frame_sync(self, seq, frame):
        self.inference_count += 1
        inference_number = self.inference_count
        log_now = self._should_log(inference_number)
        total_started = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]
        lb_img, scale, pad_x, pad_y = letterbox(frame, target=self.target_res, canvas=self._lb_canvas)
        input_data = self._prepare_input(lb_img)
        preprocess_ms = (time.perf_counter() - total_started) * 1000.0

        if log_now:
            print(
                f"\n[DEBUG #{inference_number}] frame_seq={seq} original={orig_w}x{orig_h} "
                f"letterbox_scale={scale:.6f} padding=({pad_x},{pad_y}) "
                f"input=({self._array_summary(input_data)})",
                flush=True,
            )

        npu_started = time.perf_counter()
        data = self.yolov8.nn_inference(
            [input_data],
            platform='ONNX',
            reorder='2 1 0',
            output_tensor=3,
            output_format=output_format.OUT_FORMAT_FLOAT32
        )
        npu_ms = (time.perf_counter() - npu_started) * 1000.0
        self.last_inference_ms = npu_ms

        if log_now:
            print(f"[DEBUG #{inference_number}] nn_inference_time={npu_ms:.3f} ms", flush=True)
            self._log_outputs(data, inference_number)

        postprocess_started = time.perf_counter()
        norm_boxes, curr_scores, curr_classes = self._decode_outputs(
            data, self.target_res, inference_number, log_now
        )
        if len(norm_boxes) > 0:
            curr_boxes = unletterbox_boxes(
                norm_boxes, scale, pad_x, pad_y, orig_w, orig_h, target=self.target_res
            )
        else:
            curr_boxes = np.zeros((0, 4), dtype=np.float32)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0

        completed_at = time.perf_counter()
        self.completion_times.append(completed_at)
        with self.lock:
            self.latest_detections = (seq, curr_boxes, curr_scores, curr_classes)

        if log_now or inference_number % self.result_log_every == 0:
            counts = {name: 0 for name in CLASSES}
            for class_id in curr_classes:
                class_id = int(class_id)
                if 0 <= class_id < len(CLASSES):
                    counts[CLASSES[class_id]] += 1
            count_text = ", ".join(f"{name}={counts[name]}" for name in CLASSES)
            measured_fps = (
                (len(self.completion_times) - 1)
                / (self.completion_times[-1] - self.completion_times[0])
                if len(self.completion_times) >= 2
                and self.completion_times[-1] > self.completion_times[0]
                else 0.0
            )
            detection_text = (
                ", ".join(
                    f"{CLASSES[int(class_id)]}:{float(score):.3f}"
                    for class_id, score in zip(curr_classes[:10], curr_scores[:10])
                    if 0 <= int(class_id) < len(CLASSES)
                )
                if len(curr_scores) else "none"
            )
            total_ms = (time.perf_counter() - total_started) * 1000.0
            print(
                f"[DETECTION #{inference_number}] frame={seq} | {count_text}, "
                f"total={len(curr_boxes)} | detections=[{detection_text}] | "
                f"pre={preprocess_ms:.1f}ms npu={npu_ms:.1f}ms "
                f"post={postprocess_ms:.1f}ms total={total_ms:.1f}ms "
                f"rate={measured_fps:.2f} FPS",
                flush=True,
            )

        return (seq, curr_boxes, curr_scores, curr_classes)

    def get_latest_detections(self):
        with self.lock:
            return self.latest_detections

    def submit_frame(self, seq, frame):
        # A queue of one guarantees low latency: discard an unprocessed stale
        # frame before publishing the newest raw frame.
        while True:
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        try:
            self.frame_queue.put_nowait((seq, frame))
        except Full:
            pass

    def stop(self):
        self.running = False


def run_headless_file_realtime(input_source, detector, hw_decode=True):
    """Process the same real-time file timestamps without decoding stale frames.

    The asynchronous path decodes the whole file while the NPU retains only
    the newest frame. For a slow model and ``--no-show``, seek directly to the
    frame that would be newest after the preceding inference. MJPEG supports
    exact inexpensive frame seeks; other codecs use the backend's normal seek.
    """
    reader = RTSPStreamReaderLowCPU(
        input_source,
        read_fps=30,
        hw_decode=hw_decode,
    )
    cap = reader.cap
    source_fps = float(cap.get(cv.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = 30.0
    source_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = source_frames / source_fps if source_frames > 0 else 0.0

    print("Connecting to stream/video...", flush=True)
    print("Stream connected. Processing frames...", flush=True)
    print(
        f"[HEADLESS VIDEO] mode=realtime-seek fps={source_fps:.3f} "
        f"source_frames={source_frames} duration={duration:.3f}s",
        flush=True,
    )

    timeline_started = time.perf_counter()
    target_frame = 0
    decoded_frames = 0
    last_processed_frame = 0
    eof = False

    try:
        while source_frames <= 0 or target_frame < source_frames:
            cycle_started = time.perf_counter()
            if target_frame > 0:
                if not cap.set(cv.CAP_PROP_POS_FRAMES, float(target_frame)):
                    print(
                        "[HEADLESS VIDEO] Frame seeking is unsupported by this backend; "
                        "falling back to the next decodable frame.",
                        flush=True,
                    )

            ret, frame = cap.read()
            if not ret or frame is None:
                eof = True
                break

            decoded_frames += 1
            position_after_read = int(round(cap.get(cv.CAP_PROP_POS_FRAMES)))
            actual_frame_index = (
                max(0, position_after_read - 1)
                if position_after_read > 0
                else target_frame
            )
            sequence = actual_frame_index + 1
            last_processed_frame = sequence
            detector.process_frame_sync(sequence, frame)

            if detector.process_interval > 0.0:
                remaining = detector.process_interval - (
                    time.perf_counter() - cycle_started
                )
                if remaining > 0.0:
                    time.sleep(remaining)

            # Select the frame at the current video timeline, matching the
            # previous newest-frame queue without decoding discarded frames.
            elapsed = time.perf_counter() - timeline_started
            target_frame = max(
                actual_frame_index + 1,
                int(elapsed * source_fps),
            )
            if source_frames > 0 and target_frame >= source_frames:
                eof = True
                break

    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        reader.stop()

    skipped_decode = (
        max(0, source_frames - decoded_frames)
        if source_frames > 0 else 0
    )
    print(
        f"[VIDEO SUMMARY] source_frames={source_frames} decoded_frames={decoded_frames} "
        f"decode_avoided={skipped_decode} last_processed_frame={last_processed_frame} "
        f"npu_inferences={detector.inference_count} eof={eof}",
        flush=True,
    )
    print("Done.", flush=True)
    return 0

# ─── Main Pipeline Entrypoint ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Low-CPU RTSP fire/smoke detector")
    parser.add_argument("--library", required=True, help="Path to .so library file")
    parser.add_argument("--model",   required=True, help="Path to .nb model file")
    parser.add_argument("--rtsp",    help="RTSP Stream URL")
    parser.add_argument("--video",   help="Path to local video file (.mp4, .avi, .mkv, etc.)")
    parser.add_argument("--input",   help="RTSP Stream URL or Video file path")
    parser.add_argument("--threshold", type=float, default=0.15, help="Detection threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.45, help="NMS threshold")
    parser.add_argument(
        "--score-activation",
        choices=("sigmoid", "none"),
        default="sigmoid",
        help="Class-score activation used during post-processing (default: sigmoid)",
    )
    parser.add_argument(
        "--classes",
        default="fire,other,smoke",
        help="Comma-separated class names in the exact training order (default: fire,other,smoke)",
    )
    parser.add_argument("--no-show", action="store_true", help="Run in headless mode (low CPU)")
    parser.add_argument(
        "--process-interval",
        type=float,
        default=0.100,
        help=(
            "Minimum time between inference starts in seconds "
            "(default: 0.100 = maximum 10 FPS; use 0 for unrestricted)"
        ),
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help="Maximum boxes retained after class-aware NMS (default: 100)",
    )
    parser.add_argument(
        "--result-log-every",
        type=int,
        default=None,
        help=(
            "Print one compact detection result every N inferences "
            "(default: every inference with --no-show, otherwise every 10)"
        ),
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help=(
            "Maximum display-frame width only; inference still uses the original frame "
            "(default: 1280, use 0 to disable display resizing)"
        ),
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=0.0,
        help=(
            "Optional display-only FPS limit; capture and detection remain unchanged "
            "(default: 0 uses every source frame)"
        ),
    )
    parser.add_argument(
        "--software-decode",
        action="store_true",
        help="Disable the automatic hardware-video-decoder request",
    )
    parser.add_argument("--debug", action="store_true", help="Print full model, NPU, tensor and detection diagnostics")
    parser.add_argument("--log-every", type=int, default=1, help="In debug mode, print every Nth inference (default: 1)")
    args = parser.parse_args()

    input_source = args.video or args.rtsp or args.input
    if not input_source:
        parser.error("Please specify an input stream or file using --video, --rtsp, or --input")

    global NUM_CLS, LISTSIZE, CLASSES, OBJ_THRESH, NMS_THRESH, MAX_DETECTIONS
    global SCORE_ACTIVATION
    class_names = tuple(name.strip() for name in args.classes.split(",") if name.strip())
    if len(class_names) != 3:
        parser.error("This model is three-class; --classes must contain exactly three names")
    CLASSES = class_names
    NUM_CLS = len(CLASSES)
    LISTSIZE = NUM_CLS + 64
    OBJ_THRESH = args.threshold
    NMS_THRESH = args.nms_threshold
    MAX_DETECTIONS = max(1, int(args.max_detections))
    SCORE_ACTIVATION = args.score_activation

    print("=== STARTING FIRE/SMOKE DETECTION PIPELINE ===", flush=True)
    print(
        f"Classes: {CLASSES} | output channels: {LISTSIZE} | threshold: {OBJ_THRESH} | "
        f"NMS: {NMS_THRESH} | head-order: {HEAD_ORDER} | activation: {SCORE_ACTIVATION}",
        flush=True,
    )
    print(f"Input source: {os.path.realpath(input_source) if os.path.isfile(str(input_source)) else input_source}", flush=True)
    result_log_every = (
        max(1, int(args.result_log_every))
        if args.result_log_every is not None
        else (1 if args.no_show else 10)
    )
    detector = DetectionWorker(
        args.model,
        args.library,
        process_interval=args.process_interval,
        debug=args.debug,
        log_every=args.log_every,
        result_log_every=result_log_every,
    )

    is_file = os.path.isfile(str(input_source))
    if is_file and args.no_show:
        detector.initialize()
        return run_headless_file_realtime(
            input_source,
            detector,
            hw_decode=not args.software_decode,
        )

    detector.start()
    if not is_file:
        if args.no_show:
            stream_fps = (
                min(30, max(1, int(round(1.0 / args.process_interval))))
                if args.process_interval > 0.0 else 30
            )
        else:
            stream_fps = 30
        reader = RTSPStreamReaderLowCPU(
            input_source,
            read_fps=stream_fps,
            hw_decode=not args.software_decode,
        ).start()
    else:
        reader = RTSPStreamReaderLowCPU(
            input_source,
            read_fps=30,
            hw_decode=not args.software_decode,
        ).start()

    print("Connecting to stream/video...", flush=True)
    for _ in range(100):
        ok, _, _ = reader.read(timeout=0.3)
        if ok: break
    else:
        sys.exit("[ERROR] Could not connect to stream/video.")

    print("Stream connected. Processing frames...", flush=True)
    if args.debug:
        cap = reader.cap
        fourcc_value = int(cap.get(cv.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
        print(
            "[VIDEO] "
            f"width={int(cap.get(cv.CAP_PROP_FRAME_WIDTH))} "
            f"height={int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))} "
            f"fps={cap.get(cv.CAP_PROP_FPS):.3f} "
            f"frames={int(cap.get(cv.CAP_PROP_FRAME_COUNT))} "
            f"codec={fourcc!r}",
            flush=True,
        )

    has_display = bool(os.environ.get("DISPLAY"))
    show_gui = not args.no_show and has_display

    if show_gui:
        try:
            window_name = "Real-Time Fire and Smoke Detection"
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            initial_width = args.display_width if args.display_width > 0 else 1280
            cv.resizeWindow(window_name, initial_width, max(1, int(initial_width * 9 / 16)))
        except Exception as e:
            print(f"[GUI WARNING] GUI Window initialization failed ({e}). Running in headless mode.")
            show_gui = False

    fps_times = deque(maxlen=30)
    last_seq = -1
    last_display_at = 0.0
    display_period = 1.0 / args.display_fps if args.display_fps > 0.0 else 0.0

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame, seq = reader.read(last_seq=last_seq, timeout=0.5)

            if not ret or frame is None:
                if is_file and reader.eof:
                    break
                continue

            if seq == last_seq:
                if is_file and reader.eof:
                    break
                continue

            last_seq = seq
            detector.submit_frame(seq, frame)

            if show_gui:
                try:
                    now = time.perf_counter()
                    if display_period > 0.0 and now - last_display_at < display_period:
                        key = cv.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        continue
                    last_display_at = now

                    # Never annotate the raw frame submitted to inference.
                    display_frame, scale_x, scale_y = prepare_display_frame(
                        frame,
                        args.display_width,
                    )
                    detections = detector.get_latest_detections()
                    draw_detections_fast(
                        display_frame,
                        detections,
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )

                    fps_times.append(t0)
                    current_fps = (
                        (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                        if len(fps_times) >= 2 and fps_times[-1] > fps_times[0]
                        else reader.read_fps
                    )
                    cv.putText(display_frame, f"Video FPS: {current_fps:.1f}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv.imshow(window_name, display_frame)

                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except Exception as e:
                    print(f"[GUI ERROR] Display exception: {e}")
                    show_gui = False

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as exc:
        print(f"\n[FATAL] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        reader.stop()
        detector.stop()
        if show_gui:
            try:
                cv.destroyAllWindows()
            except Exception:
                pass
        if is_file:
            print(
                f"[VIDEO SUMMARY] source_frames={reader.seq} "
                f"npu_inferences={detector.inference_count} eof={reader.eof}",
                flush=True,
            )
        print("Done.")

if __name__ == '__main__':
    main()
