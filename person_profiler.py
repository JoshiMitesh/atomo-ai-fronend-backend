#!/usr/bin/env python3

"""
REAL-TIME MULTI-CAMERA + STRICT ROUND-ROBIN NPU PERSON DETECTOR
===============================================================

Features:
- Continuous RTSP capture for every camera
- Latest-frame-only architecture
- Real-time GUI
- Strict NPU round robin:
      Cam1 -> Cam2 -> Cam3 -> Cam4 -> Cam5 -> repeat
- One shared KSNN/ASNN NPU worker
- No process interval
- No Python frame queue
- Same YOLO detection logic
- Same 1024 model support
- Person count per camera
- Total person count
- Total capture FPS
- Total NPU FPS
- CPU & Memory metrics
- Updated Local JSON detection output format
- Local NATS detection output
"""

# ==============================================================================
# ENVIRONMENT
# ==============================================================================

import os
import sys
import re

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

os.environ["OPENCV_FFMPEG_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

if "--no-show" not in sys.argv:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
        os.environ["DISPLAY"] = ":0"
else:
    if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"


# ==============================================================================
# IMPORTS
# ==============================================================================

import argparse
import threading
import time
import json
import asyncio
import gc

import cv2 as cv
import numpy as np
import psutil

import nats


cv.setNumThreads(1)

try:
    cv.ocl.setUseOpenCL(False)
except Exception:
    pass


# ==============================================================================
# KSNN / ASNN
# ==============================================================================

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

        sys.exit(
            "[ERROR] Neither KSNN nor ASNN module found."
        )


# ==============================================================================
# MODEL CONFIG
# ==============================================================================

NUM_CLS = 1
LISTSIZE = 65
TARGET_CLASS_ID = 0

CLASSES = (
    "person",
)


# ==============================================================================
# DETECTION CONFIG
# ==============================================================================

OBJ_THRESH = 0.28
NMS_THRESH = 0.25

CANDIDATE_GATHER_THRESH = 0.18
DFL_CERTAINTY_THRESH = 0.18

MIN_ASPECT_RATIO = 0.80
MIN_BOX_HEIGHT_PX = 30


# ==============================================================================
# DFL MATRIX
# ==============================================================================

constant_martix = np.array(
    [
        [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
            12, 13, 14, 15
        ]
    ]
).T


# ==============================================================================
# PREPROCESSING
# ==============================================================================

def letterbox(
    frame,
    target=640,
    canvas=None
):

    h, w = frame.shape[:2]

    scale = target / max(
        h,
        w
    )

    nh = int(
        h * scale
    )

    nw = int(
        w * scale
    )

    resized = cv.resize(
        frame,
        (
            nw,
            nh
        ),
        interpolation=cv.INTER_LINEAR
    )

    if canvas is None:

        canvas = np.full(
            (
                target,
                target,
                3
            ),
            114,
            dtype=np.uint8
        )

    else:

        canvas[:] = 114


    y0 = (
        target - nh
    ) // 2

    x0 = (
        target - nw
    ) // 2


    canvas[
        y0:y0 + nh,
        x0:x0 + nw
    ] = resized


    return (
        canvas,
        scale,
        x0,
        y0
    )


def unletterbox_boxes(
    boxes,
    scale,
    x0,
    y0,
    orig_w,
    orig_h,
    target
):

    if len(boxes) == 0:

        return np.zeros(
            (
                0,
                4
            ),
            dtype=np.float32
        )


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


    boxes[:, 0] = np.clip(
        boxes[:, 0],
        0,
        orig_w
    )

    boxes[:, 1] = np.clip(
        boxes[:, 1],
        0,
        orig_h
    )

    boxes[:, 2] = np.clip(
        boxes[:, 2],
        0,
        orig_w
    )

    boxes[:, 3] = np.clip(
        boxes[:, 3],
        0,
        orig_h
    )


    return boxes


def sigmoid(x):

    return 1 / (
        1 + np.exp(-x)
    )


def softmax(
    x,
    axis=0
):

    x = np.exp(
        x
        -
        np.max(
            x,
            axis=axis,
            keepdims=True
        )
    )


    return (
        x
        /
        x.sum(
            axis=axis,
            keepdims=True
        )
    )


# ==============================================================================
# YOLO POSTPROCESS
# ==============================================================================

def process(
    input_data,
    target_res=640
):

    listsize, grid_h, grid_w = (
        input_data.shape[:3]
    )


    if listsize == 144:

        class_idx = min(
            TARGET_CLASS_ID,
            79
        )

        class_logits = input_data[
            class_idx:class_idx + 1,
            :,
            :
        ]

        dfl_offset = 80

    else:

        class_logits = input_data[
            0:1,
            :,
            :
        ]

        dfl_offset = 1


    scores = sigmoid(
        class_logits
    )


    max_scores = np.max(
        scores,
        axis=0
    )


    pos = np.where(
        max_scores
        >=
        CANDIDATE_GATHER_THRESH
    )


    if len(pos[0]) == 0:

        return (
            np.empty(
                (
                    0,
                    4
                )
            ),
            np.empty(
                (
                    0,
                ),
                dtype=int
            ),
            np.empty(
                (
                    0,
                )
            )
        )


    y_indices = pos[0]
    x_indices = pos[1]


    dfl_data = input_data[
        dfl_offset:dfl_offset + 64,
        y_indices,
        x_indices
    ].T


    box_0 = softmax(
        dfl_data[:, 0:16],
        -1
    )

    box_1 = softmax(
        dfl_data[:, 16:32],
        -1
    )

    box_2 = softmax(
        dfl_data[:, 32:48],
        -1
    )

    box_3 = softmax(
        dfl_data[:, 48:64],
        -1
    )


    certainty = (
        np.max(
            box_0,
            axis=1
        )
        +
        np.max(
            box_1,
            axis=1
        )
        +
        np.max(
            box_2,
            axis=1
        )
        +
        np.max(
            box_3,
            axis=1
        )
    ) * 0.25


    valid_mask = (
        certainty
        >=
        DFL_CERTAINTY_THRESH
    )


    if not np.any(
        valid_mask
    ):

        return (
            np.empty(
                (
                    0,
                    4
                )
            ),
            np.empty(
                (
                    0,
                ),
                dtype=int
            ),
            np.empty(
                (
                    0,
                )
            )
        )


    y_indices = y_indices[
        valid_mask
    ]

    x_indices = x_indices[
        valid_mask
    ]

    box_0 = box_0[
        valid_mask
    ]

    box_1 = box_1[
        valid_mask
    ]

    box_2 = box_2[
        valid_mask
    ]

    box_3 = box_3[
        valid_mask
    ]


    cand_scores = scores[
        :,
        y_indices,
        x_indices
    ]


    cand_classes = np.full(
        (
            len(
                y_indices
            ),
        ),
        TARGET_CLASS_ID,
        dtype=int
    )


    cand_max_scores = np.max(
        cand_scores,
        axis=0
    )


    dfl_matrix = (
        constant_martix.ravel()
    )


    dfl_left = np.dot(
        box_0,
        dfl_matrix
    )

    dfl_top = np.dot(
        box_1,
        dfl_matrix
    )

    dfl_right = np.dot(
        box_2,
        dfl_matrix
    )

    dfl_bottom = np.dot(
        box_3,
        dfl_matrix
    )


    x1 = (
        x_indices
        +
        0.5
        -
        dfl_left
    ) / grid_w


    y1 = (
        y_indices
        +
        0.5
        -
        dfl_top
    ) / grid_h


    x2 = (
        x_indices
        +
        0.5
        +
        dfl_right
    ) / grid_w


    y2 = (
        y_indices
        +
        0.5
        +
        dfl_bottom
    ) / grid_h


    if TARGET_CLASS_ID == 0:

        w_norm = (
            x2 - x1
        )

        h_norm = (
            y2 - y1
        )


        aspect_ratio = (
            h_norm
            /
            (
                w_norm
                +
                1e-5
            )
        )


        h_px = (
            h_norm
            *
            target_res
        )


        geom_mask = (
            (
                aspect_ratio
                >=
                MIN_ASPECT_RATIO
            )
            &
            (
                h_px
                >=
                MIN_BOX_HEIGHT_PX
            )
        )


        if not np.any(
            geom_mask
        ):

            return (
                np.empty(
                    (
                        0,
                        4
                    )
                ),
                np.empty(
                    (
                        0,
                    ),
                    dtype=int
                ),
                np.empty(
                    (
                        0,
                    )
                )
            )


        x1 = x1[
            geom_mask
        ]

        y1 = y1[
            geom_mask
        ]

        x2 = x2[
            geom_mask
        ]

        y2 = y2[
            geom_mask
        ]


        cand_classes = (
            cand_classes[
                geom_mask
            ]
        )


        cand_max_scores = (
            cand_max_scores[
                geom_mask
            ]
        )


    boxes = np.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        axis=-1
    )


    return (
        boxes,
        cand_classes,
        cand_max_scores
    )


# ==============================================================================
# NMS
# ==============================================================================

def nms_boxes(
    boxes,
    scores
):

    if len(boxes) == 0:

        return np.array(
            [],
            dtype=int
        )


    bboxes_xywh = [
        [
            float(
                b[0] * 640
            ),

            float(
                b[1] * 640
            ),

            float(
                (
                    b[2] - b[0]
                ) * 640
            ),

            float(
                (
                    b[3] - b[1]
                ) * 640
            )
        ]
        for b in boxes
    ]


    indices = cv.dnn.NMSBoxes(
        bboxes_xywh,
        scores.tolist(),
        OBJ_THRESH,
        NMS_THRESH
    )


    if len(indices) == 0:

        return np.array(
            [],
            dtype=int
        )


    if isinstance(
        indices,
        (
            list,
            tuple
        )
    ):

        return np.array(
            indices,
            dtype=int
        ).ravel()


    return np.asarray(
        indices,
        dtype=int
    ).ravel()


# ==============================================================================
# JSON OUTPUT
# ==============================================================================

def create_detection_json(
    source,
    frame_id,
    frame_w,
    frame_h,
    boxes,
    scores,
    classes,
    timestamp=None
):

    if timestamp is None:

        timestamp = time.time()


    detections = []


    for box, score, class_id in zip(
        boxes,
        scores,
        classes
    ):

        class_id = int(
            class_id
        )


        if (
            0 <= class_id < len(CLASSES)
        ):

            class_name = (
                CLASSES[class_id]
            )

        else:

            class_name = "unknown"


        x1, y1, x2, y2 = box


        detections.append(
            {
                "class_id": class_id,

                "class_name": class_name,

                "confidence": round(
                    float(score),
                    4
                ),

                "bbox": {
                    "x1": round(float(x1), 1),
                    "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1),
                    "y2": round(float(y2), 1)
                }
            }
        )


    return {
        "action": "detection",
        "data": {
            "camera_id": source.name,
            "frame": {
                "id": int(frame_id),
                "timestamp": round(float(timestamp), 3),
                "width": int(frame_w),
                "height": int(frame_h)
            },
            "detections": detections
        }
    }


def save_detection_json(
    detection_data,
    output_dir
):

    camera_id = (
        detection_data.get("data", {})
        .get("camera_id", "unknown")
    )


    output_path = os.path.join(
        output_dir,
        f"{camera_id}.json"
    )


    temp_path = (
        output_path
        +
        ".tmp"
    )


    try:

        with open(
            temp_path,
            "w"
        ) as f:

            json.dump(
                detection_data,
                f,
                indent=2
            )


        os.replace(
            temp_path,
            output_path
        )


    except Exception as e:

        print(
            f"[JSON ERROR] {camera_id}: {e}",
            flush=True
        )


        try:

            if os.path.exists(
                temp_path
            ):

                os.remove(
                    temp_path
                )

        except Exception:

            pass


# ==============================================================================
# LOCAL NATS CLIENT
# ==============================================================================

class LocalNATS:

    def __init__(
        self,
        url="ws://127.0.0.1:9222",
        subject="person.detection"
    ):

        self.url = url

        self.subject = subject

        self.nc = None

        self.loop = None

        self.thread = None

        self.running = True

        self.ready = threading.Event()

        self.lock = threading.Lock()


    def start(self):

        self.thread = threading.Thread(
            target=self._run,
            name="Local-NATS",
            daemon=True
        )

        self.thread.start()


    def _run(self):

        try:

            asyncio.run(
                self._async_run()
            )

        except Exception as e:

            print(
                f"[NATS ERROR] {e}",
                flush=True
            )


    async def _async_run(self):

        self.loop = asyncio.get_running_loop()


        async def disconnected_cb():

            self.ready.clear()

            print(
                "[NATS] disconnected",
                flush=True
            )


        async def reconnected_cb():

            self.ready.set()

            print(
                "[NATS] reconnected",
                flush=True
            )


        async def error_cb(e):

            print(
                f"[NATS ERROR] {e}",
                flush=True
            )


        async def closed_cb():

            self.ready.clear()

            print(
                "[NATS] connection closed",
                flush=True
            )


        try:

            self.nc = await nats.connect(
                servers=[self.url],
                reconnect_time_wait=1,
                max_reconnect_attempts=-1,
                disconnected_cb=disconnected_cb,
                reconnected_cb=reconnected_cb,
                error_cb=error_cb,
                closed_cb=closed_cb
            )


            self.ready.set()


            print(
                f"[NATS] connected to {self.url}",
                flush=True
            )

            print(
                f"[NATS] subject: {self.subject}",
                flush=True
            )


            while self.running:

                await asyncio.sleep(
                    0.1
                )


        except Exception as e:

            self.ready.clear()

            print(
                f"[NATS ERROR] connection failed: {e}",
                flush=True
            )


    def publish(
        self,
        detection_data
    ):

        if not self.running:

            return False


        if not self.ready.is_set():

            return False


        try:

            payload = json.dumps(
                detection_data,
                separators=(
                    ",",
                    ":"
                )
            ).encode(
                "utf-8"
            )


            future = asyncio.run_coroutine_threadsafe(
                self.nc.publish(
                    self.subject,
                    payload
                ),
                self.loop
            )


            future.result(
                timeout=1.0
            )


            return True


        except Exception as e:

            print(
                f"[NATS PUBLISH ERROR] {e}",
                flush=True
            )

            return False


    def stop(self):

        self.running = False

        self.ready.clear()


        if (
            self.nc is not None
            and
            self.loop is not None
        ):

            try:

                future = asyncio.run_coroutine_threadsafe(
                    self.nc.close(),
                    self.loop
                )

                future.result(
                    timeout=2.0
                )

            except Exception:

                pass


        if self.thread is not None:

            try:

                self.thread.join(
                    timeout=2.0
                )

            except Exception:

                pass


# ==============================================================================
# CONTINUOUS LATEST-FRAME RTSP CAPTURE
# ==============================================================================

class LiveSource:

    def __init__(
        self,
        name,
        source
    ):

        self.name = name
        self.source = str(
            source
        )

        self.cap = None

        self.running = True
        self.connected = False

        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_frame_id = 0

        self.capture_fps = 0.0

        self.latest_boxes = np.zeros(
            (
                0,
                4
            ),
            dtype=np.float32
        )

        self.latest_scores = np.zeros(
            (
                0,
            ),
            dtype=np.float32
        )

        self.latest_classes = np.zeros(
            (
                0,
            ),
            dtype=np.int32
        )

        self.person_count = 0

        self.last_detection_frame = 0

        self._open()


        self.thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self.name}",
            daemon=True
        )

        self.thread.start()


    def _open(self):

        if self.cap is not None:

            try:
                self.cap.release()
            except Exception:
                pass


        os.environ[
            "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        ] = (
            "rtsp_transport;tcp|"
            "fflags;nobuffer|"
            "flags;low_delay|"
            "max_delay;0|"
            "reorder_queue_size;0|"
            "buffer_size;65536|"
            "threads;1"
        )


        self.cap = cv.VideoCapture(
            self.source,
            cv.CAP_FFMPEG
        )


        if self.cap is not None:

            try:

                self.cap.set(
                    cv.CAP_PROP_BUFFERSIZE,
                    1
                )

            except Exception:

                pass


        if (
            self.cap is not None
            and
            self.cap.isOpened()
        ):

            self.connected = True

            print(
                f"[{self.name}] stream connected",
                flush=True
            )

        else:

            self.connected = False

            print(
                f"[{self.name}] stream connection failed",
                flush=True
            )


    def _capture_loop(self):

        fps_count = 0

        fps_start = (
            time.monotonic()
        )


        while self.running:

            if (
                self.cap is None
                or
                not self.cap.isOpened()
            ):

                self.connected = False

                time.sleep(
                    0.5
                )

                if not self.running:
                    break

                self._open()

                continue


            ret, frame = (
                self.cap.read()
            )


            if (
                not ret
                or
                frame is None
            ):

                self.connected = False

                time.sleep(
                    0.1
                )

                self._open()

                continue


            self.connected = True


            with self.lock:

                # Latest reference only.
                # Previous frame is discarded automatically.
                self.latest_frame = frame

                self.latest_frame_id += 1


            fps_count += 1


            now = (
                time.monotonic()
            )


            elapsed = (
                now
                -
                fps_start
            )


            if elapsed >= 2.0:

                self.capture_fps = (
                    fps_count
                    /
                    elapsed
                )

                fps_count = 0

                fps_start = now


    def get_latest_frame(
        self,
        copy=False
    ):

        with self.lock:

            if self.latest_frame is None:

                return (
                    False,
                    None,
                    self.latest_frame_id
                )


            frame = (
                self.latest_frame
            )


            if copy:

                frame = (
                    frame.copy()
                )


            return (
                True,
                frame,
                self.latest_frame_id
            )


    def set_detection(
        self,
        boxes,
        scores,
        classes,
        scheduler_frame
    ):

        with self.lock:

            self.latest_boxes = (
                boxes.copy()
            )

            self.latest_scores = (
                scores.copy()
            )

            self.latest_classes = (
                classes.copy()
            )

            self.person_count = (
                len(boxes)
            )

            self.last_detection_frame = (
                scheduler_frame
            )


    def get_detection(self):

        with self.lock:

            return (
                self.latest_boxes.copy(),
                self.latest_scores.copy(),
                self.latest_classes.copy(),
                self.person_count,
                self.last_detection_frame
            )


    def release(self):

        self.running = False


        if self.cap is not None:

            try:

                self.cap.release()

            except Exception:

                pass


        try:

            self.thread.join(
                timeout=1.0
            )

        except Exception:

            pass


# ==============================================================================
# NPU DETECTOR
# ==============================================================================

class NPUDetector:

    def __init__(
        self,
        yolov8,
        target_res
    ):

        self.yolov8 = (
            yolov8
        )

        self.target_res = (
            target_res
        )

        self.inference_ms = 0.0


        self._lb_canvas = np.full(
            (
                target_res,
                target_res,
                3
            ),
            114,
            dtype=np.uint8
        )


        self.input_buf = np.empty(
            (
                3,
                target_res,
                target_res
            ),
            dtype=np.float32
        )


    def detect(
        self,
        frame,
        t_capture_ms=0.0
    ):

        t_start = time.perf_counter()

        orig_h, orig_w = (
            frame.shape[:2]
        )


        (
            lb_img,
            scale,
            pad_x,
            pad_y
        ) = letterbox(
            frame,
            target=self.target_res,
            canvas=self._lb_canvas
        )


        rgb_chw = (
            lb_img[
                :,
                :,
                ::-1
            ]
            .transpose(
                2,
                0,
                1
            )
        )


        np.multiply(
            rgb_chw,
            1.0 / 255.0,
            out=self.input_buf,
            casting="unsafe"
        )
        
        t_pre = time.perf_counter()


        start = (
            time.monotonic()
        )


        data = self.yolov8.nn_inference(
            [
                self.input_buf
            ],
            platform="ONNX",
            reorder="2 1 0",
            output_tensor=3,
            output_format=(
                output_format.OUT_FORMAT_FLOAT32
            )
        )

        t_inf = time.perf_counter()

        self.inference_ms = (
            (
                t_inf
                -
                t_pre
            )
            *
            1000.0
        )


        sorted_tensors = sorted(
            data,
            key=lambda d: d.size,
            reverse=True
        )


        g0_size = self.target_res // 8
        g1_size = self.target_res // 16
        g2_size = self.target_res // 32

        ls_0 = sorted_tensors[0].size // (g0_size * g0_size)
        ls_1 = sorted_tensors[1].size // (g1_size * g1_size)
        ls_2 = sorted_tensors[2].size // (g2_size * g2_size)

        input_data = [
            sorted_tensors[0].reshape(
                ls_0,
                g0_size,
                g0_size
            ),

            sorted_tensors[1].reshape(
                ls_1,
                g1_size,
                g1_size
            ),

            sorted_tensors[2].reshape(
                ls_2,
                g2_size,
                g2_size
            )
        ]


        boxes = []
        classes = []
        scores = []


        for g_inp in input_data:

            b, c, s = process(
                g_inp,
                target_res=self.target_res
            )


            if len(b) > 0:

                boxes.append(
                    b
                )

                classes.append(
                    c
                )

                scores.append(
                    s
                )


        if len(boxes) > 0:

            boxes = np.concatenate(
                boxes
            )

            classes = np.concatenate(
                classes
            )

            scores = np.concatenate(
                scores
            )


            keep = nms_boxes(
                boxes,
                scores
            )


            curr_boxes = unletterbox_boxes(
                boxes[
                    keep
                ],
                scale,
                pad_x,
                pad_y,
                orig_w,
                orig_h,
                target=self.target_res
            )


            curr_scores = (
                scores[
                    keep
                ]
            )


            curr_classes = (
                classes[
                    keep
                ]
            )


        else:

            curr_boxes = np.zeros(
                (
                    0,
                    4
                ),
                dtype=np.float32
            )


            curr_scores = np.zeros(
                (
                    0,
                ),
                dtype=np.float32
            )


            curr_classes = np.zeros(
                (
                    0,
                ),
                dtype=np.int32
            )


        t_post = time.perf_counter()
        
        p_metrics = {
            "capture": t_capture_ms,
            "pre": (t_pre - t_start) * 1000.0,
            "inf": (t_inf - t_pre) * 1000.0,
            "post": (t_post - t_inf) * 1000.0,
            "e2e": (t_post - t_start) * 1000.0 + t_capture_ms
        }
        
        # EXPLICIT RAM LEAK PATCH
        del data
        del sorted_tensors
        del input_data
        gc.collect()

        return (
            curr_boxes,
            curr_scores,
            curr_classes,
            p_metrics
        )


# ==============================================================================
# STRICT ROUND ROBIN NPU WORKER
# ==============================================================================

class RoundRobinNPUWorker:

    def __init__(
        self,
        sources,
        detector,
        json_output_dir,
        nats_client
    ):

        self.sources = (
            sources
        )

        self.detector = (
            detector
        )

        self.json_output_dir = (
            json_output_dir
        )

        self.nats_client = (
            nats_client
        )

        self.running = True

        self.camera_index = 0

        self.scheduler_frame = 0

        self.total_inference_fps = 0.0


        self.thread = threading.Thread(
            target=self._loop,
            name="NPU-RoundRobin",
            daemon=True
        )


    def start(self):

        self.thread.start()


    def _loop(self):

        fps_count = 0

        fps_start = (
            time.monotonic()
        )


        while self.running:

            if not self.sources:

                time.sleep(
                    0.01
                )

                continue


            source = self.sources[
                self.camera_index
            ]


            self.camera_index += 1


            if self.camera_index >= len(
                self.sources
            ):

                self.camera_index = 0


            self.scheduler_frame += 1


            scheduler_frame = (
                self.scheduler_frame
            )


            t_cap_start = time.perf_counter()
            ok, frame, frame_id = (
                source.get_latest_frame(
                    copy=False
                )
            )
            t_cap_end = time.perf_counter()
            cap_latency = (t_cap_end - t_cap_start) * 1000.0


            if (
                not ok
                or
                frame is None
            ):

                time.sleep(
                    0.001
                )

                continue


            frame_h, frame_w = frame.shape[:2]
            ts = time.time()


            print(
                (
                    f"[FRAME {scheduler_frame}] "
                    f"-> {source.name} "
                    f"(capture_frame={frame_id})"
                ),
                flush=True
            )


            try:

                (
                    boxes,
                    scores,
                    classes,
                    p_metrics
                ) = self.detector.detect(
                    frame,
                    t_capture_ms=cap_latency
                )


                source.set_detection(
                    boxes,
                    scores,
                    classes,
                    scheduler_frame
                )


                # ==============================================================
                # CREATE DETECTION JSON (UPDATED FORMAT)
                # ==============================================================

                detection_data = (
                    create_detection_json(
                        source=source,
                        frame_id=frame_id,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        boxes=boxes,
                        scores=scores,
                        classes=classes,
                        timestamp=ts
                    )
                )


                # ==============================================================
                # SAVE LOCAL JSON
                # ==============================================================

                save_detection_json(
                    detection_data=detection_data,
                    output_dir=self.json_output_dir
                )


                # ==============================================================
                # PUBLISH LOCAL NATS
                # ==============================================================

                nats_ok = (
                    self.nats_client.publish(
                        detection_data
                    )
                )


                if nats_ok:

                    print(
                        (
                            f"[NATS] "
                            f"{source.name} "
                            f"frame={frame_id} "
                            f"published"
                        ),
                        flush=True
                    )

                else:

                    print(
                        (
                            f"[NATS] "
                            f"{source.name} "
                            f"frame={frame_id} "
                            f"not published "
                            f"(NATS unavailable)"
                        ),
                        flush=True
                    )


                print(
                    (
                        f"[DETECT] "
                        f"FRAME={scheduler_frame} | "
                        f"{source.name} | "
                        f"Persons={len(boxes)} | "
                        f"NPU={self.detector.inference_ms:.1f}ms\n"
                        f"   [-] RTSP Fetch:  {p_metrics['capture']:.1f} ms\n"
                        f"   [-] Pre-process: {p_metrics['pre']:.1f} ms\n"
                        f"   [-] Inference:   {p_metrics['inf']:.1f} ms\n"
                        f"   [-] Post-process:{p_metrics['post']:.1f} ms\n"
                        f"   [=] E2E Latency: {p_metrics['e2e']:.1f} ms ({(1000.0/p_metrics['e2e']):.1f} FPS)"
                    ),
                    flush=True
                )


            except Exception as e:

                print(
                    (
                        f"[NPU ERROR] "
                        f"{source.name}: "
                        f"{e}"
                    ),
                    flush=True
                )


                time.sleep(
                    0.01
                )

                continue


            fps_count += 1


            now = (
                time.monotonic()
            )


            elapsed = (
                now
                -
                fps_start
            )


            if elapsed >= 2.0:

                self.total_inference_fps = (
                    fps_count
                    /
                    elapsed
                )

                fps_count = 0

                fps_start = now


    def stop(self):

        self.running = False


        try:

            self.thread.join(
                timeout=3.0
            )

        except Exception:

            pass


# ==============================================================================
# GUI
# ==============================================================================

def draw_detections(
    image,
    boxes,
    scores,
    cam_name,
    person_count,
    detection_frame
):

    for box, score in zip(
        boxes,
        scores
    ):

        x1, y1, x2, y2 = box


        left = max(
            0,
            int(round(x1))
        )

        top = max(
            0,
            int(round(y1))
        )

        right = min(
            image.shape[1],
            int(round(x2))
        )

        bottom = min(
            image.shape[0],
            int(round(y2))
        )


        if (
            right <= left
            or
            bottom <= top
        ):

            continue


        cv.rectangle(
            image,
            (
                left,
                top
            ),
            (
                right,
                bottom
            ),
            (
                0,
                255,
                0
            ),
            2
        )


        label = (
            f"person {float(score):.2f}"
        )


        cv.putText(
            image,
            label,
            (
                left,
                max(
                    20,
                    top - 5
                )
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (
                0,
                255,
                0
            ),
            2,
            cv.LINE_AA
        )


    banner = (
        f"{cam_name} | Persons: {person_count} | "
        f"DetectFrame: {detection_frame}"
    )


    cv.rectangle(
        image,
        (
            5,
            5
        ),
        (
            420,
            40
        ),
        (
            0,
            0,
            0
        ),
        -1
    )


    cv.putText(
        image,
        banner,
        (
            12,
            30
        ),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            0,
            255,
            0
        ),
        2,
        cv.LINE_AA
    )


def build_grid_mosaic(
    frames,
    target_w=1280,
    target_h=720
):

    num_frames = len(
        frames
    )


    if num_frames == 0:

        return np.zeros(
            (
                target_h,
                target_w,
                3
            ),
            dtype=np.uint8
        )


    if num_frames == 1:

        return cv.resize(
            frames[0],
            (
                target_w,
                target_h
            ),
            interpolation=cv.INTER_NEAREST
        )


    cols = int(
        np.ceil(
            np.sqrt(
                num_frames
            )
        )
    )


    rows = int(
        np.ceil(
            num_frames
            /
            cols
        )
    )


    cell_w = (
        target_w
        //
        cols
    )


    cell_h = (
        target_h
        //
        rows
    )


    grid = np.zeros(
        (
            target_h,
            target_w,
            3
        ),
        dtype=np.uint8
    )


    for index, frame in enumerate(
        frames
    ):

        row = (
            index // cols
        )

        col = (
            index % cols
        )


        resized = cv.resize(
            frame,
            (
                cell_w,
                cell_h
            ),
            interpolation=cv.INTER_NEAREST
        )


        y1 = (
            row * cell_h
        )

        x1 = (
            col * cell_w
        )


        y2 = min(
            y1 + cell_h,
            target_h
        )

        x2 = min(
            x1 + cell_w,
            target_w
        )


        grid[
            y1:y2,
            x1:x2
        ] = resized[
            :y2-y1,
            :x2-x1
        ]


    return grid


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Real-time multi-camera "
            "strict round-robin NPU detector"
        )
    )


    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True
    )


    parser.add_argument(
        "--library",
        required=True
    )


    parser.add_argument(
        "--model",
        required=True
    )


    parser.add_argument(
        "--threshold",
        type=float,
        default=0.28
    )


    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.25
    )


    parser.add_argument(
        "--no-show",
        action="store_true"
    )


    parser.add_argument(
        "--display-width",
        type=int,
        default=1280
    )


    parser.add_argument(
        "--display-height",
        type=int,
        default=720
    )


    parser.add_argument(
        "--display-fps",
        type=float,
        default=25.0
    )


    args = parser.parse_args()


    global OBJ_THRESH
    global NMS_THRESH


    OBJ_THRESH = (
        args.threshold
    )

    NMS_THRESH = (
        args.nms_threshold
    )


    show_gui = (
        not args.no_show
    )


    # ==========================================================================
    # LOCAL JSON OUTPUT
    # ==========================================================================

    JSON_OUTPUT_DIR = (
        "/tmp/person_detections"
    )


    os.makedirs(
        JSON_OUTPUT_DIR,
        exist_ok=True
    )


    # ==========================================================================
    # LOCAL NATS CONFIGURATION
    # ==========================================================================

    NATS_URL = (
        "nats://127.0.0.1:4222"
    )

    NATS_SUBJECT = (
        "person.detection"
    )


    print()

    print(
        "=============================================================="
    )

    print(
        " REAL-TIME + STRICT ROUND-ROBIN PERSON DETECTOR"
    )

    print(
        "=============================================================="
    )

    print(
        f"NPU              : {NPU_KIND}"
    )

    print(
        f"Cameras          : {len(args.inputs)}"
    )

    print(
        f"Threshold        : {OBJ_THRESH}"
    )

    print(
        f"NMS              : {NMS_THRESH}"
    )

    print(
        f"GUI              : {'ON' if show_gui else 'OFF'}"
    )

    print(
        "Capture          : CONTINUOUS"
    )

    print(
        "Capture queue    : LATEST FRAME ONLY"
    )

    print(
        "NPU scheduler    : STRICT ROUND ROBIN"
    )

    print(
        "Process interval : DISABLED"
    )

    print(
        f"JSON output      : {JSON_OUTPUT_DIR}"
    )

    print(
        f"NATS             : {NATS_URL}"
    )

    print(
        f"NATS subject     : {NATS_SUBJECT}"
    )

    print(
        "==============================================================",
        flush=True
    )


    # ==========================================================================
    # LOCAL NATS
    # ==========================================================================

    nats_client = LocalNATS(
        url=NATS_URL,
        subject=NATS_SUBJECT
    )


    nats_client.start()


    # ==========================================================================
    # SOURCES
    # ==========================================================================

    sources = []


    for index, url in enumerate(
        args.inputs
    ):

        source = LiveSource(
            name=f"Cam-{index + 1}",
            source=url
        )


        sources.append(
            source
        )


    # ==========================================================================
    # NPU
    # ==========================================================================

    model_path = os.path.abspath(
        args.model
    )

    library_path = os.path.abspath(
        args.library
    )


    if NPU_KIND == "KSNN":

        yolov8 = NPU_API(
            "VIM3"
        )

    else:

        yolov8 = NPU_API(
            "Electron"
        )


    print(
        f"[{NPU_KIND}] Initializing NPU...",
        flush=True
    )


    result = yolov8.nn_init(
        library=library_path,
        model=model_path,
        level=0
    )


    print(
        (
            f"[{NPU_KIND}] "
            f"nn_init result: "
            f"{result}"
        ),
        flush=True
    )


    # Auto-detect resolution from model filename
    match = re.search(r'(256|320|384|416|448|480|512|544|576|608|640|672|704|736|768|800|832|864|896|928|960|992|1024|1056|1088|1120|1152|1184|1216|1248|1280)', os.path.basename(model_path))
    if match:
        target_res = int(match.group(1))
    else:
        target_res = 640  # Fallback


    print(
        (
            f"[{NPU_KIND}] "
            f"model resolution: "
            f"{target_res}x{target_res}"
        ),
        flush=True
    )


    detector = NPUDetector(
        yolov8=yolov8,
        target_res=target_res
    )


    worker = RoundRobinNPUWorker(
        sources=sources,
        detector=detector,
        json_output_dir=JSON_OUTPUT_DIR,
        nats_client=nats_client
    )


    worker.start()


    # ==========================================================================
    # GUI
    # ==========================================================================

    window_name = (
        "Real-Time Multi-Camera Person Detection"
    )


    if show_gui:

        try:

            cv.namedWindow(
                window_name,
                cv.WINDOW_NORMAL
            )


            cv.resizeWindow(
                window_name,
                args.display_width,
                args.display_height
            )


            print(
                "[GUI] Press q to quit.",
                flush=True
            )


        except Exception as e:

            print(
                f"[GUI WARNING] {e}",
                flush=True
            )

            show_gui = False


    # ==========================================================================
    # METRICS & DISPLAY LOOP
    # ==========================================================================

    process_proc = psutil.Process(
        os.getpid()
    )


    process_proc.cpu_percent(
        interval=None
    )


    display_interval = (
        1.0
        /
        max(
            1.0,
            args.display_fps
        )
    )


    next_display = (
        time.monotonic()
    )


    # ==========================================================================
    # MAIN LOOP
    # ==========================================================================

    try:

        while True:

            now = (
                time.monotonic()
            )


            if show_gui and now >= next_display:

                next_display = (
                    now
                    +
                    display_interval
                )


                gui_frames = []


                for source in sources:

                    (
                        ok,
                        frame,
                        frame_id
                    ) = source.get_latest_frame(
                        copy=True
                    )


                    if (
                        not ok
                        or
                        frame is None
                    ):

                        frame = np.zeros(
                            (
                                360,
                                640,
                                3
                            ),
                            dtype=np.uint8
                        )


                        cv.putText(
                            frame,
                            (
                                f"{source.name}: "
                                f"NO SIGNAL"
                            ),
                            (
                                20,
                                50
                            ),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (
                                0,
                                0,
                                255
                            ),
                            2
                        )


                    else:

                        (
                            boxes,
                            scores,
                            classes,
                            person_count,
                            detection_frame
                        ) = source.get_detection()


                        draw_detections(
                            frame,
                            boxes,
                            scores,
                            source.name,
                            person_count,
                            detection_frame
                        )


                    gui_frames.append(
                        frame
                    )


                mosaic = build_grid_mosaic(
                    gui_frames,
                    args.display_width,
                    args.display_height
                )


                total_people = sum(
                    source.person_count
                    for source in sources
                )

                total_cap_fps = sum(
                    source.capture_fps
                    for source in sources
                )

                cpu_pct = process_proc.cpu_percent()
                mem_pct = psutil.virtual_memory().percent

                summary_banner = (
                    f"TOTAL PERSONS: {total_people} | "
                    f"NPU FPS: {worker.total_inference_fps:.1f} | "
                    f"CAP FPS: {total_cap_fps:.1f} | "
                    f"CPU: {cpu_pct:.1f}% | "
                    f"MEM: {mem_pct:.1f}%"
                )


                cv.rectangle(
                    mosaic,
                    (
                        0,
                        0
                    ),
                    (
                        mosaic.shape[1],
                        34
                    ),
                    (
                        0,
                        0,
                        0
                    ),
                    -1
                )

                cv.putText(
                    mosaic,
                    summary_banner,
                    (
                        12,
                        24
                    ),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (
                        0,
                        255,
                        255
                    ),
                    2,
                    cv.LINE_AA
                )

                cv.imshow(
                    window_name,
                    mosaic
                )

                key = cv.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    print(
                        "[GUI] Exit requested by user.",
                        flush=True
                    )
                    break

            else:

                time.sleep(
                    0.02
                )

    except KeyboardInterrupt:

        print(
            "\n[MAIN] Interrupted by user (Ctrl+C).",
            flush=True
        )

    except Exception as e:

        print(
            f"\n[MAIN ERROR] Unexpected error: {e}",
            flush=True
        )

    finally:

        print(
            "[MAIN] Shutting down detector system...",
            flush=True
        )

        if worker is not None:
            worker.stop()

        if nats_client is not None:
            nats_client.stop()

        for source in sources:
            source.release()

        if show_gui:
            cv.destroyAllWindows()

        print(
            "[MAIN] System shutdown complete.",
            flush=True
        )


if __name__ == "__main__":
    main()
