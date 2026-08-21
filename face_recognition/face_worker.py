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
    def __init__(self, modelPath: str, confThreshold: float = 0.80, try_npu: bool = True):
        self.try_npu = try_npu
        # Default input size, will be set dynamically per frame
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
        self._disType = disType  # 0 cosine, 1 norml2
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

# Multi-camera threads and stop events dictionaries
stream_threads = {}
stream_stop_events = {}
streams_lock = threading.Lock()



def load_models():
    global detector, recog, yunet_path, sface_path
    # Production/offline mode: never contact Hugging Face at runtime.
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
    
    cpu_detector.setInputSize((img.shape[1], img.shape[0]))
    faces = cpu_detector.infer(img)
    if faces.shape[0] == 0:
        return None
    
    # Get the largest face by bounding box area (w * h) to avoid enrolling background faces
    areas = faces[:, 2] * faces[:, 3]
    best_idx = np.argmax(areas)
    best_face = faces[best_idx]
    
    # Enforce quality checks for enrollment templates
    x, y, w, h = best_face[:4].astype(int)
    conf = best_face[-1]
    
    if enforce_quality:
        if conf < 0.60:
            raise ValueError(f"Face detection confidence is too low ({conf:.2f} < 0.60). Use a clearer photo.")
        if w < 50 or h < 50:
            raise ValueError(f"Face is too small ({w}x{h} < 50x50 pixels). Use a closer photo of the face.")
        
    feat = cpu_recog.infer(img, best_face[:-1])
    if feat is None:
        raise ValueError("Could not extract face embedding feature.")
    return feat.flatten().tolist()

def compare_face_features(feat, threshold, dis_type, thread_recog):
    with candidates_lock:
        local_candidates = list(candidates)
        
    if not local_candidates:
        return None, -1.0
        
    cand_scores = []
    for cand in local_candidates:
        cand_id = cand.get("person_id")
        cand_name = cand.get("name")
        embs = cand.get("embeddings", [])
        
        if not embs:
            continue
            
        scores = []
        for emb_list in embs:
            emb_arr = np.array(emb_list, dtype=np.float32).reshape(1, -1)
            score = thread_recog.score(feat, emb_arr)
            scores.append(score)
            
        # Sort scores: Cosine -> descending (highest first); L2 -> ascending (lowest first)
        scores.sort(reverse=(dis_type == 0))
        
        # Use nearest-neighbor match (highest score among all enrolled templates of this person)
        best_cand_score = scores[0]
            
        cand_scores.append({
            "person_id": cand_id,
            "name": cand_name,
            "score": best_cand_score
        })
        
    if not cand_scores:
        return None, -1.0
        
    # Sort candidates by score
    if dis_type == 0: # Cosine: higher score first
        cand_scores.sort(key=lambda x: x["score"], reverse=True)
    else: # L2: lower score first
        cand_scores.sort(key=lambda x: x["score"])
        
    best_cand = cand_scores[0]
    best_score = best_cand["score"]
    
    is_match = False
    if dis_type == 0: # Cosine
        is_match = best_score >= threshold
    else: # L2
        is_match = best_score <= threshold
                    
    if is_match:
        return {"person_id": best_cand["person_id"], "name": best_cand["name"]}, best_score
    else:
        return None, best_score


def crop_and_save_face(img, box, crops_dir):
    h_img, w_img = img.shape[:2]
    x, y, w, h = box.astype(int)
    
    # 60% margin padding to show clear face with surrounding head/shoulders context
    pad_w = int(w * 0.60)
    pad_h = int(h * 0.60)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)
    
    if x2 > x1 and y2 > y1:
        crop = img[y1:y2, x1:x2]
        
        # Upscale small crops to a minimum of 150x150 for consistent display quality
        crop_h, crop_w = crop.shape[:2]
        min_size = 150
        if crop_h < min_size or crop_w < min_size:
            scale_factor = max(min_size / crop_w, min_size / crop_h)
            new_w = int(crop_w * scale_factor)
            new_h = int(crop_h * scale_factor)
            crop = cv.resize(crop, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # to enhance brightness and contrast for dark/distant camera crops
        try:
            lab = cv.cvtColor(crop, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            enhanced = cv.merge([l, a, b])
            crop = cv.cvtColor(enhanced, cv.COLOR_LAB2BGR)
        except Exception:
            pass  # If enhancement fails, save original crop
        
        crop_filename = f"crop_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
        crop_path = os.path.join(crops_dir, crop_filename)
        cv.imwrite(crop_path, crop, [cv.IMWRITE_JPEG_QUALITY, 90])
        return crop_filename
    return None

def process_video_for_enrollment(video_path, crops_dir, person_id=None):
    import shutil
    
    # Create a unique temporary directory for frame images
    timestamp = int(time.time() * 1000)
    temp_frames_dir = os.path.join(os.path.dirname(video_path), f"temp_frames_{timestamp}")
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    accepted_faces = []
    try:
        log(f"Extracting frames using system FFmpeg to {temp_frames_dir}...")
        # Extract 2 frames per second to get ~40 high-quality frames from a 20s scan
        cmd = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-vf', 'fps=2',
            '-pix_fmt', 'yuv420p',
            os.path.join(temp_frames_dir, 'frame_%04d.jpg')
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Read the extracted frame images
        frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.jpg')])
        log(f"Extracted {len(frame_files)} frames. Running face detection on CPU...")
        
        cpu_detector = YuNet(yunet_path, try_npu=False)
        cpu_recog = SFace(sface_path, try_npu=False)
        
        # Set input size once before loop
        det_w = 640
        det_h = 480
        if len(frame_files) > 0:
            first_frame_path = os.path.join(temp_frames_dir, frame_files[0])
            first_frame = cv.imread(first_frame_path)
            if first_frame is not None:
                orig_h, orig_w = first_frame.shape[:2]
                det_h = int(orig_h * (det_w / orig_w))
                cpu_detector.setInputSize((det_w, det_h))
        
        for f_file in frame_files:
            f_path = os.path.join(temp_frames_dir, f_file)
            frame = cv.imread(f_path)
            if frame is None:
                continue
                
            orig_h, orig_w = frame.shape[:2]
            
            # Downscale frame for extremely fast CPU detection (takes ~25ms instead of 800ms)
            small_frame = cv.resize(frame, (det_w, det_h))
            faces = cpu_detector.infer(small_frame)
            
            if faces is not None and faces.shape[0] > 0:
                # Find the largest face bounding box
                areas = faces[:, 2] * faces[:, 3]
                best_idx = np.argmax(areas)
                best_face = faces[best_idx]
                
                # Scale coordinates back to original high-resolution size
                scale_x = orig_w / det_w
                scale_y = orig_h / det_h
                
                scaled_face = best_face.copy()
                scaled_face[0] = best_face[0] * scale_x
                scaled_face[1] = best_face[1] * scale_y
                scaled_face[2] = best_face[2] * scale_x
                scaled_face[3] = best_face[3] * scale_y
                
                # Scale landmarks
                for idx in range(5):
                    scaled_face[4 + idx*2] = best_face[4 + idx*2] * scale_x
                    scaled_face[4 + idx*2 + 1] = best_face[4 + idx*2 + 1] * scale_y
                
                w, h = scaled_face[2], scaled_face[3]
                conf = best_face[-1]
                
                # Minimum face size of 50x50 and detection confidence >= 0.60
                if conf >= 0.60 and w >= 50 and h >= 50:
                    # Run SFace on original high-resolution image using scaled landmarks
                    feat = cpu_recog.infer(frame, scaled_face[:-1])
                    if feat is not None:
                        # Crop and save from original high-resolution frame
                        crop_filename = crop_and_save_face(frame, scaled_face[:4], crops_dir)
                        if crop_filename:
                            embedding_list = feat.flatten().tolist()
                            
                            # Filter out duplicate/similar face angles (Cosine similarity >= 0.85)
                            # to keep the candidate list small, clean, and extremely fast to recognize.
                            is_unique = True
                            feat_arr = np.array(embedding_list, dtype=np.float32).reshape(1, -1)
                            for accepted in accepted_faces:
                                accepted_emb = np.array(accepted["embedding"], dtype=np.float32).reshape(1, -1)
                                sim = cpu_recog.score(feat_arr, accepted_emb)
                                if sim >= 0.85:
                                    is_unique = False
                                    break
                                    
                            if is_unique:
                                accepted_faces.append({
                                    "filename": crop_filename,
                                    "embedding": embedding_list
                                })
                                
                                # Emit real-time enrollment event
                                if person_id:
                                    sys.stdout.write(json.dumps({
                                        "event": "video_enroll_face",
                                        "person_id": person_id,
                                        "filename": crop_filename,
                                        "embedding": embedding_list
                                    }) + "\n")
                                    sys.stdout.flush()
                            
                        # Cap total enrollment count per video scan to 500 templates
                        if len(accepted_faces) >= 500:
                            break
                            
    except Exception as ex:
        log(f"Error extracting video frames: {str(ex)}")
        raise ex
    finally:
        # Clean up temporary frames directory
        try:
            shutil.rmtree(temp_frames_dir)
        except Exception as ec:
            log(f"Failed to remove temp directory {temp_frames_dir}: {str(ec)}")
            
    return accepted_faces

def process_single_image(img_path, threshold, dis_type, crops_dir):
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
        
    cpu_detector = YuNet(yunet_path, try_npu=False)
    cpu_recog = SFace(sface_path, disType=dis_type, try_npu=False)
    
    cpu_detector.setInputSize((img.shape[1], img.shape[0]))
    faces = cpu_detector.infer(img)
    
    results = []
    for f in faces:
        box = f[:4]
        conf = f[-1]
        
        is_known = False
        match = None
        score = 0.0
        feat = None
        
        # Only run SFace recognition on high-confidence face detections to prevent false matches
        if conf >= 0.60:
            feat = cpu_recog.infer(img, f[:-1])
            if feat is not None:
                match, score = compare_face_features(feat, threshold, dis_type, cpu_recog)
                is_known = match is not None
            
        crop_filename = crop_and_save_face(img, box, crops_dir)
        
        results.append({
            "box": box.astype(int).tolist(),
            "score": score,
            "match": match,
            "is_known": is_known,
            "crop_filename": crop_filename,
            "embedding": feat.flatten().tolist() if feat is not None else None
        })
        
    return results

class VideoGrabber(threading.Thread):
    def __init__(self, rtsp_url, camera_id):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.running = True
        self.latest_frame = None
        self.need_frame = True
        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event()
        self.process = None
        self.daemon = True

    def run(self):
        log(f"[{self.camera_id}] Starting FFmpeg raw reader for RTSP...")
        
        # Enforce constant resolution of 960x540 BGR24
        width = 960
        height = 540
        frame_size = width * height * 3
        
        reconnect_delay = 1.0
        while self.running:
            try:
                cmd = [
                    'ffmpeg',
                    '-allowed_media_types', 'video', # Force video track only
                    '-rtsp_transport', 'tcp',
                    '-fflags', 'nobuffer',         # Disable input buffers
                    '-flags', 'low_delay',         # Force low latency flags
                    '-probesize', '100000',         # Minimal stream analysis probe
                    '-analyzeduration', '0',        # 0 analyze duration
                    '-threads', '1',              # Limit decoding threads to 1
                    '-i', self.rtsp_url,
                    '-vf', 'scale=960:540',        # Force consistent 960x540 output resolution
                    '-f', 'image2pipe',
                    '-pix_fmt', 'bgr24',
                    '-vcodec', 'rawvideo',
                    '-an',
                    '-'
                ]
                
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=frame_size * 2
                )
                
                log(f"[{self.camera_id}] FFmpeg subprocess pipe connected.")
                reconnect_delay = 1.0
                
                import select
                while self.running:
                    raw_data = self.process.stdout.read(frame_size)
                    if not raw_data or len(raw_data) < frame_size:
                        log(f"[{self.camera_id}] FFmpeg pipe EOF / connection dropped. Reconnecting...")
                        break
                        
                    # Drain any accumulated stale frames in pipe buffer to prevent video lag drift over hours
                    try:
                        while True:
                            r, _, _ = select.select([self.process.stdout], [], [], 0.0)
                            if r:
                                more = self.process.stdout.read(frame_size)
                                if more and len(more) == frame_size:
                                    raw_data = more
                                else:
                                    break
                            else:
                                break
                    except Exception:
                        pass
                        
                    if self.need_frame:
                        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                        with self.frame_lock:
                            self.latest_frame = frame.copy()
                        self.frame_event.set()
                        self.need_frame = False
                        
                # Cleanup subprocess
                if self.process:
                    try:
                        self.process.stdout.close()
                    except:
                        pass
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=1.0)
                    except:
                        try:
                            self.process.kill()
                        except:
                            pass
                    self.process = None
                    
            except Exception as e:
                log(f"[{self.camera_id}] Exception in VideoGrabber thread loop: {str(e)}")
                if self.process:
                    try: self.process.terminate()
                    except: pass
                    self.process = None
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 10.0)

    def get_frame(self, timeout=0.2):
        got_new = self.frame_event.wait(timeout)
        if got_new:
            self.frame_event.clear()
            with self.frame_lock:
                frame = self.latest_frame
            self.need_frame = True
            return frame
        else:
            return None

    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.terminate()
            except:
                pass

def rtsp_stream_processor(camera_id, camera_name, rtsp_url, threshold, dis_type, crops_dir, 
                          line_crossing_enabled, line_y, line_direction, line_x_start, line_x_end, stop_event):
    log(f"[{camera_id}] Starting camera stream thread: {camera_name} (Line Crossing: {line_crossing_enabled}, Y: {line_y}, Dir: {line_direction}, X: {line_x_start}-{line_x_end})")
    
    # Separate thread-local models to prevent race conditions on inference
    thread_detector = YuNet(yunet_path)
    thread_recog = SFace(sface_path, disType=dis_type)
    

        

        
    grabber = VideoGrabber(rtsp_url, camera_id)
    grabber.start()
    
    last_event_time = {}
    det_width = 640
    recent_events = []

    # Face tracking state (line-crossing mode)
    tracked_faces = []
    next_track_id = 0

    # Face tracking state (standard mode) — lets a visible face keep the
    # same event_uuid across frames instead of being gated to one detection
    # every 3 seconds. Box position updates every frame (instant/continuous);
    # recognition is only re-queued periodically per-track to control SFace cost.
    std_tracked_faces = []
    next_std_track_id = 0
    STD_TRACK_MAX_DIST_FACTOR = 0.10   # fraction of frame width used as match radius
    STD_TRACK_STALE_S = 0.5            # drop a track if unseen for this long
    STD_RECOGNIZE_RETRY_S = 3.0        # re-attempt recognition for an unknown track this often
    STD_EVENT_COOLDOWN_S = 4.0         # log an event at most once every 4s per track
    STD_MAX_RECOG_ATTEMPTS = 3         # max recognition retries per track

    last_frame_time = 0.0
    
    try:
        while not stop_event.is_set():
            frame = grabber.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
                
            now = time.time()
            dt = now - last_frame_time if last_frame_time > 0.0 else 0.2
            last_frame_time = now
            
            h_img, w_img = frame.shape[:2]
            
            # Optimization: Resize frame for YuNet face detection (reduces CPU by up to 90% for 1080p feeds)
            if w_img > det_width:
                scale = det_width / float(w_img)
                det_h = int(h_img * scale)
                det_frame = cv.resize(frame, (det_width, det_h))
            else:
                scale = 1.0
                det_frame = frame
                det_h, det_width = h_img, w_img
                
            thread_detector.setInputSize((det_width, det_h))
            faces = thread_detector.infer(det_frame)
            
            detected_faces_data = []
            now = time.time()
            
            if line_crossing_enabled:
                # ---------------------------------------------------------
                # Line Crossing Face Detection & Tracking Mode
                # ---------------------------------------------------------
                current_tracked = []
                matched_ids = set()
                y_line = h_img * line_y
                x_start = w_img * line_x_start
                x_end = w_img * line_x_end
                
                for f in faces:
                    f_orig = f.copy()
                    if scale != 1.0:
                        f_orig[:14] = f_orig[:14] / scale
                        
                    box = f_orig[:4]
                    x, y, w, h = box.astype(int)
                    conf = f_orig[-1]
                    
                    if w < 15 or h < 15:
                        continue
                        
                    cx = x + w / 2
                    cy = y + h / 2
                    
                    best_match = None
                    best_dist = float('inf')
                    # Scale search radius dynamically based on frame-step time (dt) to prevent track splits on slow hardware
                    base_max_dist = w_img * 0.08
                    max_dist = min(base_max_dist * (dt / 0.2), w_img * 0.20)
                    
                    for tf in tracked_faces:
                        if tf["id"] in matched_ids:
                            continue
                        # If a track has already crossed and exited the line, do not match it with a new face above the line
                        if tf["crossed"] and cy <= y_line:
                            continue
                        tx, ty = tf["last_center"]
                        # Prevent hijacking: if track is crossed, don't match with a face behind the track's movement direction
                        if tf["crossed"] and len(tf["ys"]) >= 2:
                            track_dir = tf["ys"][-1] - tf["ys"][0]
                            if track_dir > 0 and cy < ty: # Moving down, face is above track
                                continue
                            elif track_dir < 0 and cy > ty: # Moving up, face is below track
                                continue
                        dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                        if dist < max_dist and dist < best_dist:
                            best_dist = dist
                            best_match = tf
                            
                    if best_match is not None:
                        matched_ids.add(best_match["id"])
                        prev_cy = best_match["last_center"][1]
                        best_match["last_bbox"] = [x, y, w, h]
                        best_match["last_center"] = (cx, cy)
                        best_match["last_seen"] = now
                        best_match["ys"].append(cy)
                        if len(best_match["ys"]) > 10:
                            best_match["ys"].pop(0)
                            
                        crossed_trigger = False
                        if not best_match["crossed"]:
                            prev_y = best_match["ys"][-2] if len(best_match["ys"]) >= 2 else best_match.get("initial_y", cy)
                            curr_y = cy

                            cross_in = (prev_y <= y_line and curr_y > y_line) or (best_match.get("initial_y", cy) <= y_line and curr_y > y_line)
                            cross_out = (prev_y >= y_line and curr_y < y_line) or (best_match.get("initial_y", cy) >= y_line and curr_y < y_line)

                            if (x_start <= cx <= x_end):
                                if line_direction == 'in' and cross_in:
                                    crossed_trigger = True
                                elif line_direction == 'out' and cross_out:
                                    crossed_trigger = True
                                elif line_direction == 'both' and (cross_in or cross_out):
                                    crossed_trigger = True

                        if crossed_trigger:
                            # Try recognizing if detection confidence is sufficient (>= 0.50)
                            if conf >= 0.50:
                                best_match["crossed"] = True
                                log(f"[{camera_id}] Track #{best_match['id']} crossed line Y={int(y_line)} (X span: {int(x_start)}-{int(x_end)}) in direction: {line_direction}")
                                
                                event_uuid = f"{camera_id}_{int(now*1000)}_{best_match['id']}"
                                crop_filename = crop_and_save_face(frame, box, crops_dir)
                                
                                # Emit DETECT immediately!
                                sys.stdout.write(json.dumps({
                                    "event": "stream_detect",
                                    "camera_id": camera_id,
                                    "camera_name": camera_name,
                                    "event_uuid": event_uuid,
                                    "box": [int(x), int(y), int(w), int(h)],
                                    "crop_filename": crop_filename
                                }) + "\n")
                                sys.stdout.flush()
                                
                                # Queue recognition task to the background thread asynchronously
                                recognition_queue.put({
                                    "camera_id": camera_id,
                                    "camera_name": camera_name,
                                    "event_uuid": event_uuid,
                                    "frame": frame.copy(),
                                    "f_orig": f_orig.copy(),
                                    "crop_filename": crop_filename,
                                    "threshold": threshold,
                                    "dis_type": dis_type,
                                    "recog_model": thread_recog
                                })
                        current_tracked.append(best_match)
                    else:
                        new_tf = {
                            "id": next_track_id,
                            "last_bbox": [x, y, w, h],
                            "last_center": (cx, cy),
                            "crossed": False,
                            "last_seen": now,
                            "ys": [cy],
                            "initial_y": cy
                        }
                        next_track_id += 1
                        current_tracked.append(new_tf)
                        
                # Preserve unmatched tracks for a short period (0.3 seconds) to handle brief dropouts
                for tf in tracked_faces:
                    if tf["id"] not in matched_ids:
                        if now - tf["last_seen"] < 0.3:
                            current_tracked.append(tf)
                            
                tracked_faces = current_tracked
            else:
                # ---------------------------------------------------------
                # Standard Mode (rate-limited event logging per track)
                # ---------------------------------------------------------
                current_std_tracked = []
                matched_std_ids = set()
                max_dist = w_img * STD_TRACK_MAX_DIST_FACTOR

                for f in faces:
                    f_orig = f.copy()
                    if scale != 1.0:
                        f_orig[:14] = f_orig[:14] / scale

                    box = f_orig[:4]
                    x, y, w, h = box.astype(int)
                    conf = f_orig[-1]

                    if w < 15 or h < 15 or conf < 0.60:
                        continue

                    cx = x + w / 2
                    cy = y + h / 2

                    best_match = None
                    best_dist = float('inf')
                    for tf in std_tracked_faces:
                        if tf["id"] in matched_std_ids:
                            continue
                        tx, ty = tf["last_center"]
                        dist = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
                        if dist < max_dist and dist < best_dist:
                            best_dist = dist
                            best_match = tf

                    if best_match is None:
                        # New face entering the frame — start a fresh track
                        event_uuid = f"{camera_id}_{int(now*1000)}_std{next_std_track_id}"
                        best_match = {
                            "id": next_std_track_id,
                            "event_uuid": event_uuid,
                            "last_center": (cx, cy),
                            "last_seen": now,
                            "recognized": False,
                            "last_recognize_attempt": 0.0,
                            "last_event_time": 0.0,
                            "crop_filename": None,
                            "recognize_attempts": 0,
                        }
                        next_std_track_id += 1

                    matched_std_ids.add(best_match["id"])
                    best_match["last_center"] = (cx, cy)
                    best_match["last_seen"] = now

                    # Rate-limit event logging: emit DETECT and crop image ONLY once per track arrival,
                    # or every 15s if person remains in view continuously
                    should_log_event = (now - best_match.get("last_event_time", 0.0)) >= STD_EVENT_COOLDOWN_S

                    if should_log_event:
                        best_match["last_event_time"] = now
                        crop_filename = crop_and_save_face(frame, box, crops_dir)
                        best_match["crop_filename"] = crop_filename
                        event_uuid = f"{camera_id}_{int(now*1000)}_std{best_match['id']}"
                        best_match["event_uuid"] = event_uuid

                        # Emit DETECT event with current track recognition state
                        sys.stdout.write(json.dumps({
                            "event": "stream_detect",
                            "camera_id": camera_id,
                            "camera_name": camera_name,
                            "event_uuid": event_uuid,
                            "box": [int(x), int(y), int(w), int(h)],
                            "crop_filename": crop_filename,
                            "is_known": best_match.get("recognized", False),
                            "match": best_match.get("match", None),
                            "score": best_match.get("score", 0.0)
                        }) + "\n")
                        sys.stdout.flush()

                    crop_filename = best_match.get("crop_filename")

                    # Queue recognition for new track or retry up to 3 times if unrecognized
                    recognize_attempts = best_match.get("recognize_attempts", 0)
                    should_recognize = (
                        not best_match["recognized"]
                        and recognize_attempts < STD_MAX_RECOG_ATTEMPTS
                        and (now - best_match["last_recognize_attempt"]) >= STD_RECOGNIZE_RETRY_S
                    )
                    if should_recognize:
                        if not crop_filename:
                            crop_filename = crop_and_save_face(frame, box, crops_dir)
                            best_match["crop_filename"] = crop_filename

                        best_match["last_recognize_attempt"] = now
                        best_match["recognize_attempts"] = recognize_attempts + 1
                        recognition_queue.put({
                            "camera_id": camera_id,
                            "camera_name": camera_name,
                            "event_uuid": best_match["event_uuid"],
                            "frame": frame.copy(),
                            "f_orig": f_orig.copy(),
                            "crop_filename": crop_filename,
                            "threshold": threshold,
                            "dis_type": dis_type,
                            "recog_model": thread_recog,
                            "std_track": best_match,
                        })

                    current_std_tracked.append(best_match)

                # Preserve unmatched tracks briefly to survive a missed frame or two
                for tf in std_tracked_faces:
                    if tf["id"] not in matched_std_ids and (now - tf["last_seen"]) < STD_TRACK_STALE_S:
                        current_std_tracked.append(tf)

                std_tracked_faces = current_std_tracked

            # Regulate thread loop rate adaptively to conserve CPU on ARM64 while preventing frame skips
            if len(tracked_faces) > 0 or len(std_tracked_faces) > 0:
                # High-speed tracking mode: sleep minimal time to capture every frame
                time.sleep(0.01)
            else:
                # Low-power standby mode: sleep 0.25s (~4 FPS) to save CPU when scene is empty
                time.sleep(0.25)
            
    except Exception as e:
        log(f"[{camera_id}] Error in processor loop: {str(e)}")
    finally:
        grabber.stop()
        log(f"[{camera_id}] Camera stream thread terminated.")

def recognition_worker_thread():
    while True:
        try:
            task = recognition_queue.get()
            if task is None:
                break
                
            camera_id = task["camera_id"]
            camera_name = task["camera_name"]
            event_uuid = task["event_uuid"]
            frame = task["frame"]
            f_orig = task["f_orig"]
            crop_filename = task["crop_filename"]
            threshold = task["threshold"]
            dis_type = task["dis_type"]
            recog_model = task["recog_model"]
            
            # Run SFace inference on the background thread
            feat = recog_model.infer(frame, f_orig[:-1])
            is_known = False
            match = None
            score = 0.0
            if feat is not None:
                match, score = compare_face_features(feat, threshold, dis_type, recog_model)
                is_known = match is not None

            # For standard-mode continuous tracks: stop re-queuing recognition
            # once a face is confirmed known (saves SFace calls); unknown faces
            # keep retrying every STD_RECOGNIZE_RETRY_S in case of a bad angle.
            std_track = task.get("std_track")
            if std_track is not None and is_known:
                std_track["recognized"] = True
                std_track["match"] = match
                std_track["score"] = score

            # Emit RECOGNIZE event immediately once SFace completes
            sys.stdout.write(json.dumps({
                "event": "stream_recognize",
                "camera_id": camera_id,
                "event_uuid": event_uuid,
                "score": score,
                "match": match,
                "is_known": is_known,
                "crop_filename": crop_filename,
                "embedding": feat.flatten().tolist() if feat is not None else None
            }) + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            log(f"Error in recognition worker thread: {str(e)}")
        finally:
            recognition_queue.task_done()

def main():
    global candidates, stream_threads, stream_stop_events
    load_models()
    
    # Start background asynchronous recognition worker thread
    threading.Thread(target=recognition_worker_thread, daemon=True).start()
    
    sys.stdout.write(json.dumps({"event": "ready"}) + "\n")
    sys.stdout.flush()
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            req = json.loads(line.strip())
            cmd = req.get("cmd")
            
            if cmd == "extract_embedding":
                img_path = req.get("img_path")
                enforce_quality = req.get("enforce_quality", True)
                try:
                    emb = extract_best_face_embedding(img_path, enforce_quality=enforce_quality)
                    if emb:
                        res = {"status": "success", "embedding": emb}
                    else:
                        res = {"status": "error", "message": "No face detected in image."}
                except Exception as e:
                    res = {"status": "error", "message": str(e)}
                sys.stdout.write(json.dumps({"cmd": "extract_embedding", "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "process_video_enrollment":
                video_path = req.get("video_path")
                crops_dir = req.get("crops_dir")
                person_id = req.get("person_id")
                try:
                    faces = process_video_for_enrollment(video_path, crops_dir, person_id)
                    res = {"status": "success", "faces": faces}
                except Exception as e:
                    res = {"status": "error", "message": str(e)}
                sys.stdout.write(json.dumps({"cmd": "process_video_enrollment", "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "recognize_image":
                img_path = req.get("img_path")
                local_candidates = req.get("candidates", [])
                threshold = req.get("threshold", 0.60)
                dis_type = req.get("dis_type", 0)
                crops_dir = req.get("crops_dir", ".")
                
                with candidates_lock:
                    candidates = local_candidates
                    
                try:
                    faces = process_single_image(img_path, threshold, dis_type, crops_dir)
                    res = {"status": "success", "faces": faces}
                except Exception as e:
                    res = {"status": "error", "message": str(e)}
                sys.stdout.write(json.dumps({"cmd": "recognize_image", "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "start_stream":
                camera_id = req.get("camera_id")
                camera_name = req.get("camera_name", "Unnamed Camera")
                rtsp_url = req.get("rtsp_url")
                local_candidates = req.get("candidates", [])
                threshold = req.get("threshold", 0.60)
                dis_type = req.get("dis_type", 0)
                crops_dir = req.get("crops_dir", ".")
                line_crossing_enabled = req.get("line_crossing_enabled", False)
                line_y = req.get("line_y", 0.6)
                line_direction = req.get("line_direction", "in")
                line_x_start = req.get("line_x_start", 0.0)
                line_x_end = req.get("line_x_end", 1.0)
                
                with candidates_lock:
                    candidates = local_candidates
                
                with streams_lock:
                    # Stop stream if running
                    if camera_id in stream_threads:
                        log(f"Restarting camera {camera_id}...")
                        stream_stop_events[camera_id].set()
                        stream_threads[camera_id].join(timeout=2.0)
                        
                    stop_event = threading.Event()
                    stream_stop_events[camera_id] = stop_event
                    
                    thread = threading.Thread(
                        target=rtsp_stream_processor,
                        args=(camera_id, camera_name, rtsp_url, threshold, dis_type, crops_dir, 
                              line_crossing_enabled, line_y, line_direction, line_x_start, line_x_end, stop_event),
                        daemon=True
                    )
                    stream_threads[camera_id] = thread
                    thread.start()
                
                res = {"status": "success", "message": f"Camera stream thread {camera_id} started."}
                sys.stdout.write(json.dumps({"cmd": "start_stream", "camera_id": camera_id, "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "stop_stream":
                camera_id = req.get("camera_id")
                
                with streams_lock:
                    if camera_id in stream_threads:
                        stream_stop_events[camera_id].set()
                        
                        # Join in background to not block the main reading thread
                        def cleanup_thread(cid):
                            if cid in stream_threads:
                                stream_threads[cid].join(timeout=2.0)
                                del stream_threads[cid]
                                if cid in stream_stop_events:
                                    del stream_stop_events[cid]
                        
                        threading.Thread(target=cleanup_thread, args=(camera_id,), daemon=True).start()
                        res = {"status": "success", "message": f"Camera {camera_id} stopped."}
                    else:
                        res = {"status": "success", "message": "Camera stream was not running."}
                        
                sys.stdout.write(json.dumps({"cmd": "stop_stream", "camera_id": camera_id, "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "update_candidates":
                local_candidates = req.get("candidates", [])
                with candidates_lock:
                    candidates = local_candidates
                res = {"status": "success", "message": "Candidates synced across all stream threads."}
                sys.stdout.write(json.dumps({"cmd": "update_candidates", "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "add_template":
                person_id = req.get("person_id")
                embedding = req.get("embedding")
                with candidates_lock:
                    found = False
                    for cand in candidates:
                        if cand.get("person_id") == person_id:
                            if "embeddings" not in cand:
                                cand["embeddings"] = []
                            cand["embeddings"].append(embedding)
                            found = True
                            break
                    if not found:
                        candidates.append({
                            "person_id": person_id,
                            "name": req.get("name", "Unknown"),
                            "embeddings": [embedding]
                        })
                res = {"status": "success", "message": "Embedding appended successfully."}
                sys.stdout.write(json.dumps({"cmd": "add_template", "response": res}) + "\n")
                sys.stdout.flush()
                
            else:
                res = {"status": "error", "message": f"Unknown command: {cmd}"}
                sys.stdout.write(json.dumps({"cmd": cmd, "response": res}) + "\n")
                sys.stdout.flush()
                
        except Exception as e:
            log(f"Error reading/processing command: {str(e)}")
            res = {"status": "error", "message": str(e)}
            sys.stdout.write(json.dumps({"response": res}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
