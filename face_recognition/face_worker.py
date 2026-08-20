import sys
import os
import queue
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import json
import time
import base64
import random
import threading
import subprocess
import cv2 as cv
cv.setNumThreads(2)
import numpy as np
from huggingface_hub import hf_hub_download

npu_lock = threading.Lock()

# Print to stderr for debugging so stdout remains clean JSON
def log(msg):
    sys.stderr.write(f"[Python Worker] {msg}\n")
    sys.stderr.flush()

# NPU / TIM-VX Configuration Constants
DNN_BACKEND_TIMVX = getattr(cv.dnn, 'DNN_BACKEND_TIMVX', 7)
DNN_TARGET_NPU = getattr(cv.dnn, 'DNN_TARGET_NPU', 9)

def configure_dnn_target(net):
    try:
        net.setPreferableBackend(DNN_BACKEND_TIMVX)
        net.setPreferableTarget(DNN_TARGET_NPU)
        log("ONNX model configured to use TIM-VX NPU.")
    except Exception as e:
        log(f"Failed to set TIM-VX NPU backend for ONNX model, using CPU fallback: {str(e)}")

class YuNet:
    def __init__(self, modelPath: str, confThreshold: float = 0.90, try_npu: bool = True):
        self.try_npu = try_npu
        # Keep 0.90 detector confidence to avoid false detections. Distant-face
        # recall is improved by preserving more input pixels, not by lowering
        # this threshold.
        self._model = None
        if try_npu:
            try:
                self._model = cv.FaceDetectorYN.create(
                    modelPath, "", (320, 320), confThreshold, 0.3, 5000,
                    DNN_BACKEND_TIMVX, DNN_TARGET_NPU
                )
                log("YuNet initialized successfully on TIM-VX NPU!")
            except Exception as e:
                log(f"Failed to load YuNet on TIM-VX NPU, falling back to CPU: {str(e)}")
                self._model = None

        if self._model is None:
            self._model = cv.FaceDetectorYN.create(
                modelPath, "", (320, 320), confThreshold, 0.3, 5000, 0, 0
            )

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        if self.try_npu:
            with npu_lock:
                faces = self._model.detect(image)
        else:
            faces = self._model.detect(image)
        return np.empty((0, 15), dtype=np.float32) if faces[1] is None else faces[1]

class SFace:
    def __init__(self, modelPath: str, disType: int = 0, try_npu: bool = False):
        self.try_npu = try_npu
        self._disType = disType
        self._model = None
        if try_npu:
            try:
                self._model = cv.FaceRecognizerSF.create(
                    modelPath, "", DNN_BACKEND_TIMVX, DNN_TARGET_NPU
                )
                log("SFace initialized successfully on TIM-VX NPU!")
            except Exception as e:
                log(f"Failed to load SFace on TIM-VX NPU, falling back to CPU: {str(e)}")
                self._model = None

        if self._model is None:
            self._model = cv.FaceRecognizerSF.create(modelPath, "", 0, 0)

    def infer(self, image, face_bbox_landmarks_etc):
        try:
            if image is None or image.size == 0:
                return None
            if not np.isfinite(face_bbox_landmarks_etc).all():
                return None
            if self.try_npu:
                with npu_lock:
                    aligned = self._model.alignCrop(image, face_bbox_landmarks_etc.astype(np.float32))
                    if aligned is None or aligned.size == 0 or aligned.shape[0] == 0 or aligned.shape[1] == 0:
                        return None
                    feat = self._model.feature(aligned)
                    if feat is None or feat.size == 0:
                        return None
                    return feat
            else:
                aligned = self._model.alignCrop(image, face_bbox_landmarks_etc.astype(np.float32))
                if aligned is None or aligned.size == 0 or aligned.shape[0] == 0 or aligned.shape[1] == 0:
                    return None
                feat = self._model.feature(aligned)
                if feat is None or feat.size == 0:
                    return None
                return feat
        except Exception as e:
            log(f"Error in SFace.infer: {str(e)}")
            return None

    def score(self, feat1, feat2) -> float:
        if feat1 is None or feat2 is None:
            return 0.0
        try:
            return float(self._model.match(feat1, feat2, self._disType))
        except Exception as e:
            log(f"Error in SFace.score: {str(e)}")
            return 0.0

# Global variables for model paths and main thread instances
yunet_path = ""
sface_path = ""
detector = None
recog = None

candidates = []
candidates_lock = threading.Lock()

recognition_queue = queue.Queue()
recog_lock = threading.Lock()

stream_threads = {}
stream_stop_events = {}
streams_lock = threading.Lock()


def load_models():
    global detector, recog, yunet_path, sface_path
    # Production/offline mode: never contact Hugging Face at runtime.
    # The models have already been downloaded on this device; resolve them from
    # the local HF cache only. If the cache is missing, fail immediately instead
    # of waiting on an unavailable internet connection.
    log("Loading face models from local cache (offline mode)...")
    try:
        yunet_path = hf_hub_download(
            "opencv/face_detection_yunet",
            "face_detection_yunet_2023mar_int8.onnx",
            local_files_only=True,
        )
        sface_path = hf_hub_download(
            "opencv/face_recognition_sface",
            "face_recognition_sface_2021dec_int8.onnx",
            local_files_only=True,
        )
        log("Local face models found. Loading main thread instances...")
        detector = YuNet(yunet_path)
        recog = SFace(sface_path)
        log("Models loaded in memory.")
    except Exception as e:
        log("OFFLINE MODEL ERROR: required YuNet/SFace model is not available in the local Hugging Face cache.")
        log(f"Details: {str(e)}")
        sys.exit(1)


def extract_best_face_embedding(img_path, enforce_quality=True):
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    cpu_detector = YuNet(yunet_path, try_npu=False)
    cpu_recog = SFace(sface_path, try_npu=False)
    h, w = img.shape[:2]
    cpu_detector.setInputSize((w, h))
    faces = cpu_detector.infer(img)
    if len(faces) == 0:
        raise ValueError("No face detected in image")

    best_face = max(faces, key=lambda f: float(f[-1]))
    if enforce_quality:
        fw, fh = float(best_face[2]), float(best_face[3])
        if min(fw, fh) < 70:
            raise ValueError("Face is too small for reliable enrollment")

    feat = cpu_recog.infer(img, best_face)
    if feat is None:
        raise ValueError("Could not extract face embedding")
    return feat.flatten().astype(float).tolist()

# Remaining worker code is intentionally loaded from the repository version below.
# This marker is replaced during deployment if this file is regenerated.
