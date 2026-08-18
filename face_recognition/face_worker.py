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
    log("Downloading models...")
    try:
        yunet_path = hf_hub_download("opencv/face_detection_yunet", "face_detection_yunet_2023mar_int8.onnx")
        sface_path = hf_hub_download("opencv/face_recognition_sface", "face_recognition_sface_2021dec_int8.onnx")
        log("Models downloaded successfully. Loading main thread instances...")
        detector = YuNet(yunet_path)
        recog = SFace(sface_path)
        log("Models loaded in memory.")
    except Exception as e:
        log(f"Error loading models: {str(e)}")
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

    areas = faces[:, 2] * faces[:, 3]
    best_idx = np.argmax(areas)
    best_face = faces[best_idx]

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

        scores.sort(reverse=(dis_type == 0))
        best_cand_score = scores[0]
        cand_scores.append({"person_id": cand_id, "name": cand_name, "score": best_cand_score})

    if not cand_scores:
        return None, -1.0

    if dis_type == 0:
        cand_scores.sort(key=lambda x: x["score"], reverse=True)
    else:
        cand_scores.sort(key=lambda x: x["score"])

    best_cand = cand_scores[0]
    best_score = best_cand["score"]
    is_match = best_score >= threshold if dis_type == 0 else best_score <= threshold
    if is_match:
        return {"person_id": best_cand["person_id"], "name": best_cand["name"]}, best_score
    return None, best_score


def crop_and_save_face(img, box, crops_dir):
    h_img, w_img = img.shape[:2]
    x, y, w, h = box.astype(int)
    pad_w = int(w * 0.60)
    pad_h = int(h * 0.60)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)

    if x2 > x1 and y2 > y1:
        crop = img[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]
        min_size = 150
        if crop_h < min_size or crop_w < min_size:
            scale_factor = max(min_size / crop_w, min_size / crop_h)
            crop = cv.resize(crop, (int(crop_w * scale_factor), int(crop_h * scale_factor)), interpolation=cv.INTER_CUBIC)
        try:
            lab = cv.cvtColor(crop, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            crop = cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2BGR)
        except Exception:
            pass
        crop_filename = f"crop_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
        cv.imwrite(os.path.join(crops_dir, crop_filename), crop, [cv.IMWRITE_JPEG_QUALITY, 90])
        return crop_filename
    return None

# Preserve the rest of the original implementation below. The runtime changes
# relevant to crowded/distant scenes are in VideoGrabber and rtsp_stream_processor.

def process_video_for_enrollment(video_path, crops_dir, person_id=None):
    import shutil
    timestamp = int(time.time() * 1000)
    temp_frames_dir = os.path.join(os.path.dirname(video_path), f"temp_frames_{timestamp}")
    os.makedirs(temp_frames_dir, exist_ok=True)
    accepted_faces = []
    try:
        log(f"Extracting frames using system FFmpeg to {temp_frames_dir}...")
        cmd = ['ffmpeg','-y','-i',video_path,'-vf','fps=2','-pix_fmt','yuv420p',os.path.join(temp_frames_dir,'frame_%04d.jpg')]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.jpg')])
        log(f"Extracted {len(frame_files)} frames. Running face detection on CPU...")
        cpu_detector = YuNet(yunet_path, try_npu=False)
        cpu_recog = SFace(sface_path, try_npu=False)
        det_w = 640
        det_h = 480
        if frame_files:
            first_frame = cv.imread(os.path.join(temp_frames_dir, frame_files[0]))
            if first_frame is not None:
                orig_h, orig_w = first_frame.shape[:2]
                det_h = int(orig_h * (det_w / orig_w))
                cpu_detector.setInputSize((det_w, det_h))
        for f_file in frame_files:
            frame = cv.imread(os.path.join(temp_frames_dir, f_file))
            if frame is None: continue
            orig_h, orig_w = frame.shape[:2]
            small_frame = cv.resize(frame, (det_w, det_h))
            faces = cpu_detector.infer(small_frame)
            if faces is not None and faces.shape[0] > 0:
                areas = faces[:,2] * faces[:,3]
                best_face = faces[np.argmax(areas)]
                scale_x, scale_y = orig_w/det_w, orig_h/det_h
                scaled_face = best_face.copy()
                scaled_face[0:4:2] *= scale_x
                scaled_face[1:4:2] *= scale_y
                for idx in range(5):
                    scaled_face[4+idx*2] *= scale_x
                    scaled_face[5+idx*2] *= scale_y
                w,h = scaled_face[2],scaled_face[3]
                conf = best_face[-1]
                if conf >= 0.60 and w >= 50 and h >= 50:
                    feat = cpu_recog.infer(frame, scaled_face[:-1])
                    if feat is not None:
                        crop_filename = crop_and_save_face(frame, scaled_face[:4], crops_dir)
                        if crop_filename:
                            embedding_list = feat.flatten().tolist()
                            is_unique = True
                            feat_arr = np.array(embedding_list,dtype=np.float32).reshape(1,-1)
                            for accepted in accepted_faces:
                                accepted_emb=np.array(accepted["embedding"],dtype=np.float32).reshape(1,-1)
                                if cpu_recog.score(feat_arr,accepted_emb)>=0.85:
                                    is_unique=False; break
                            if is_unique:
                                accepted_faces.append({"filename":crop_filename,"embedding":embedding_list})
                                if person_id:
                                    sys.stdout.write(json.dumps({"event":"video_enroll_face","person_id":person_id,"filename":crop_filename,"embedding":embedding_list})+"\n");sys.stdout.flush()
                        if len(accepted_faces)>=500: break
    finally:
        try: shutil.rmtree(temp_frames_dir)
        except Exception: pass
    return accepted_faces

def process_single_image(img_path, threshold, dis_type, crops_dir):
    img=cv.imread(img_path)
    if img is None: raise ValueError(f"Could not read image: {img_path}")
    cpu_detector=YuNet(yunet_path,try_npu=False);cpu_recog=SFace(sface_path,disType=dis_type,try_npu=False)
    cpu_detector.setInputSize((img.shape[1],img.shape[0]));faces=cpu_detector.infer(img)
    results=[]
    for f in faces:
        box=f[:4];conf=f[-1];is_known=False;match=None;score=0.0;feat=None
        if conf>=0.60:
            feat=cpu_recog.infer(img,f[:-1])
            if feat is not None:
                match,score=compare_face_features(feat,threshold,dis_type,cpu_recog);is_known=match is not None
        crop_filename=crop_and_save_face(img,box,crops_dir)
        results.append({"box":box.astype(int).tolist(),"score":score,"match":match,"is_known":is_known,"crop_filename":crop_filename,"embedding":feat.flatten().tolist() if feat is not None else None})
    return results

class VideoGrabber(threading.Thread):
    def __init__(self,rtsp_url,camera_id):
        super().__init__();self.rtsp_url=rtsp_url;self.camera_id=camera_id;self.running=True;self.latest_frame=None;self.need_frame=True;self.frame_lock=threading.Lock();self.frame_event=threading.Event();self.process=None;self.daemon=True
    def run(self):
        log(f"[{self.camera_id}] Starting FFmpeg raw reader for RTSP...")
        # Keep a 960x540 low-latency working frame. YuNet now receives all of
        # these pixels instead of another 640px downscale.
        width,height=960,540;frame_size=width*height*3;reconnect_delay=1.0
        while self.running:
            try:
                cmd=['ffmpeg','-allowed_media_types','video','-rtsp_transport','tcp','-fflags','nobuffer','-flags','low_delay','-probesize','100000','-analyzeduration','0','-threads','1','-i',self.rtsp_url,'-vf','scale=960:540','-f','image2pipe','-pix_fmt','bgr24','-vcodec','rawvideo','-an','-']
                self.process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=frame_size*2)
                log(f"[{self.camera_id}] FFmpeg subprocess pipe connected.");reconnect_delay=1.0
                import select
                while self.running:
                    raw_data=self.process.stdout.read(frame_size)
                    if not raw_data or len(raw_data)<frame_size:
                        log(f"[{self.camera_id}] FFmpeg pipe EOF / connection dropped. Reconnecting...");break
                    try:
                        while True:
                            r,_,_=select.select([self.process.stdout],[],[],0.0)
                            if not r: break
                            more=self.process.stdout.read(frame_size)
                            if more and len(more)==frame_size: raw_data=more
                            else: break
                    except Exception: pass
                    if self.need_frame:
                        frame=np.frombuffer(raw_data,dtype=np.uint8).reshape((height,width,3))
                        with self.frame_lock:self.latest_frame=frame.copy()
                        self.frame_event.set();self.need_frame=False
                if self.process:
                    try:self.process.stdout.close()
                    except:pass
                    try:self.process.terminate();self.process.wait(timeout=1.0)
                    except:
                        try:self.process.kill()
                        except:pass
                    self.process=None
            except Exception as e:
                log(f"[{self.camera_id}] Exception in VideoGrabber thread loop: {str(e)}")
                if self.process:
                    try:self.process.terminate()
                    except:pass
                    self.process=None
                time.sleep(reconnect_delay);reconnect_delay=min(reconnect_delay*2,10.0)
    def get_frame(self,timeout=0.2):
        got_new=self.frame_event.wait(timeout)
        if got_new:
            self.frame_event.clear()
            with self.frame_lock:frame=self.latest_frame
            self.need_frame=True;return frame
        return None
    def stop(self):
        self.running=False
        if self.process:
            try:self.process.terminate()
            except:pass

def rtsp_stream_processor(camera_id,camera_name,rtsp_url,threshold,dis_type,crops_dir,line_crossing_enabled,line_y,line_direction,line_x_start,line_x_end,stop_event):
    log(f"[{camera_id}] Starting camera stream thread: {camera_name} (Line Crossing: {line_crossing_enabled}, Y: {line_y}, Dir: {line_direction}, X: {line_x_start}-{line_x_end})")
    thread_detector=YuNet(yunet_path)
    thread_recog=SFace(sface_path,disType=dis_type)
    grabber=VideoGrabber(rtsp_url,camera_id);grabber.start()
    tracked_faces=[];next_track_id=0;std_tracked_faces=[];next_std_track_id=0
    STD_TRACK_MAX_DIST_FACTOR=0.10;STD_TRACK_STALE_S=0.8;STD_RECOGNIZE_RETRY_S=2.0;STD_EVENT_COOLDOWN_S=4.0;STD_MAX_RECOG_ATTEMPTS=5
    last_frame_time=0.0
    # Full 960px detection preserves ~50% more linear face detail than the old
    # 640px pass. This is the key change for a wide room containing many people.
    det_width=960
    try:
        while not stop_event.is_set():
            frame=grabber.get_frame()
            if frame is None: time.sleep(0.05);continue
            now=time.time();dt=now-last_frame_time if last_frame_time>0 else 0.2;last_frame_time=now
            h_img,w_img=frame.shape[:2]
            if w_img>det_width:
                scale=det_width/float(w_img);det_h=max(1,int(h_img*scale));det_frame=cv.resize(frame,(det_width,det_h),interpolation=cv.INTER_LINEAR)
            else:
                scale=1.0;det_frame=frame;det_h,det_w=h_img,w_img;det_width_current=det_w
            if scale!=1.0: det_width_current=det_width
            thread_detector.setInputSize((det_width_current,det_h))
            faces=thread_detector.infer(det_frame)
            now=time.time()
            if line_crossing_enabled:
                current_tracked=[];matched_ids=set();y_line=h_img*line_y;x_start=w_img*line_x_start;x_end=w_img*line_x_end
                for f in faces:
                    f_orig=f.copy()
                    if scale!=1.0:f_orig[:14]=f_orig[:14]/scale
                    box=f_orig[:4];x,y,w,h=box.astype(int);conf=f_orig[-1]
                    if w<12 or h<12:continue
                    cx=x+w/2;cy=y+h/2;best_match=None;best_dist=float('inf');base_max_dist=w_img*0.08;max_dist=min(base_max_dist*(dt/0.2),w_img*0.20)
                    for tf in tracked_faces:
                        if tf["id"] in matched_ids:continue
                        if tf["crossed"] and cy<=y_line:continue
                        tx,ty=tf["last_center"];dist=np.sqrt((cx-tx)**2+(cy-ty)**2)
                        if dist<max_dist and dist<best_dist:best_dist=dist;best_match=tf
                    if best_match is not None:
                        matched_ids.add(best_match["id"]);best_match["last_bbox"]=[x,y,w,h];best_match["last_center"]=(cx,cy);best_match["last_seen"]=now;best_match["ys"].append(cy);best_match["ys"]=best_match["ys"][-10:]
                        prev_y=best_match["ys"][-2] if len(best_match["ys"])>=2 else best_match.get("initial_y",cy);cross_in=(prev_y<=y_line and cy>y_line);cross_out=(prev_y>=y_line and cy<y_line);trigger=(x_start<=cx<=x_end) and ((line_direction=='in' and cross_in) or (line_direction=='out' and cross_out) or (line_direction=='both' and (cross_in or cross_out)))
                        if trigger and not best_match["crossed"] and conf>=0.50:
                            best_match["crossed"]=True;event_uuid=f"{camera_id}_{int(now*1000)}_{best_match['id']}";crop_filename=crop_and_save_face(frame,box,crops_dir)
                            sys.stdout.write(json.dumps({"event":"stream_detect","camera_id":camera_id,"camera_name":camera_name,"event_uuid":event_uuid,"box":[int(x),int(y),int(w),int(h)],"crop_filename":crop_filename})+"\n");sys.stdout.flush()
                            recognition_queue.put({"camera_id":camera_id,"camera_name":camera_name,"event_uuid":event_uuid,"frame":frame.copy(),"f_orig":f_orig.copy(),"crop_filename":crop_filename,"threshold":threshold,"dis_type":dis_type,"recog_model":thread_recog})
                        current_tracked.append(best_match)
                    else:
                        current_tracked.append({"id":next_track_id,"last_bbox":[x,y,w,h],"last_center":(cx,cy),"crossed":False,"last_seen":now,"ys":[cy],"initial_y":cy});next_track_id+=1
                for tf in tracked_faces:
                    if tf["id"] not in matched_ids and now-tf["last_seen"]<0.5:current_tracked.append(tf)
                tracked_faces=current_tracked
            else:
                current_std_tracked=[];matched_std_ids=set();max_dist=w_img*STD_TRACK_MAX_DIST_FACTOR
                for f in faces:
                    f_orig=f.copy()
                    if scale!=1.0:f_orig[:14]=f_orig[:14]/scale
                    box=f_orig[:4];x,y,w,h=box.astype(int);conf=f_orig[-1]
                    # YuNet already enforces 0.90 at detector level. Keep tiny-face
                    # rejection modest so distant valid faces are not thrown away.
                    if w<12 or h<12:continue
                    cx=x+w/2;cy=y+h/2;best_match=None;best_dist=float('inf')
                    for tf in std_tracked_faces:
                        if tf["id"] in matched_std_ids:continue
                        tx,ty=tf["last_center"];dist=np.sqrt((cx-tx)**2+(cy-ty)**2)
                        if dist<max_dist and dist<best_dist:best_dist=dist;best_match=tf
                    if best_match is None:
                        event_uuid=f"{camera_id}_{int(now*1000)}_std{next_std_track_id}";best_match={"id":next_std_track_id,"event_uuid":event_uuid,"last_center":(cx,cy),"last_seen":now,"recognized":False,"last_recognize_attempt":0.0,"last_event_time":0.0,"crop_filename":None,"recognize_attempts":0};next_std_track_id+=1
                    matched_std_ids.add(best_match["id"]);best_match["last_center"]=(cx,cy);best_match["last_seen"]=now
                    should_log_event=(now-best_match.get("last_event_time",0.0))>=STD_EVENT_COOLDOWN_S
                    if should_log_event:
                        best_match["last_event_time"]=now;crop_filename=crop_and_save_face(frame,box,crops_dir);best_match["crop_filename"]=crop_filename;event_uuid=f"{camera_id}_{int(now*1000)}_std{best_match['id']}";best_match["event_uuid"]=event_uuid
                        sys.stdout.write(json.dumps({"event":"stream_detect","camera_id":camera_id,"camera_name":camera_name,"event_uuid":event_uuid,"box":[int(x),int(y),int(w),int(h)],"crop_filename":crop_filename,"is_known":best_match.get("recognized",False),"match":best_match.get("match"),"score":best_match.get("score",0.0)})+"\n");sys.stdout.flush()
                    crop_filename=best_match.get("crop_filename");attempts=best_match.get("recognize_attempts",0)
                    should_recognize=not best_match["recognized"] and attempts<STD_MAX_RECOG_ATTEMPTS and (now-best_match["last_recognize_attempt"])>=STD_RECOGNIZE_RETRY_S
                    if should_recognize:
                        if not crop_filename:crop_filename=crop_and_save_face(frame,box,crops_dir);best_match["crop_filename"]=crop_filename
                        best_match["last_recognize_attempt"]=now;best_match["recognize_attempts"]=attempts+1
                        recognition_queue.put({"camera_id":camera_id,"camera_name":camera_name,"event_uuid":best_match["event_uuid"],"frame":frame.copy(),"f_orig":f_orig.copy(),"crop_filename":crop_filename,"threshold":threshold,"dis_type":dis_type,"recog_model":thread_recog,"std_track":best_match})
                    current_std_tracked.append(best_match)
                for tf in std_tracked_faces:
                    if tf["id"] not in matched_std_ids and now-tf["last_seen"]<STD_TRACK_STALE_S:current_std_tracked.append(tf)
                std_tracked_faces=current_std_tracked
            # Full-resolution YuNet is heavier than the old 640px pass, so use a
            # modest loop delay while keeping enough temporal coverage for tracking.
            time.sleep(0.03 if (tracked_faces or std_tracked_faces) else 0.12)
    except Exception as e:
        log(f"[{camera_id}] Error in processor loop: {str(e)}")
    finally:
        grabber.stop();log(f"[{camera_id}] Camera stream thread terminated.")

def recognition_worker_thread():
    while True:
        try:
            task=recognition_queue.get()
            if task is None:break
            camera_id=task["camera_id"];event_uuid=task["event_uuid"];frame=task["frame"];f_orig=task["f_orig"];crop_filename=task["crop_filename"];threshold=task["threshold"];dis_type=task["dis_type"];recog_model=task["recog_model"]
            feat=recog_model.infer(frame,f_orig[:-1]);is_known=False;match=None;score=0.0
            if feat is not None:
                match,score=compare_face_features(feat,threshold,dis_type,recog_model);is_known=match is not None
            std_track=task.get("std_track")
            if std_track is not None and is_known:
                std_track["recognized"]=True;std_track["match"]=match;std_track["score"]=score
            sys.stdout.write(json.dumps({"event":"stream_recognize","camera_id":camera_id,"event_uuid":event_uuid,"score":score,"match":match,"is_known":is_known,"crop_filename":crop_filename,"embedding":feat.flatten().tolist() if feat is not None else None})+"\n");sys.stdout.flush()
        except Exception as e:log(f"Error in recognition worker thread: {str(e)}")
        finally:recognition_queue.task_done()

def main():
    global candidates,stream_threads,stream_stop_events
    load_models();threading.Thread(target=recognition_worker_thread,daemon=True).start();sys.stdout.write(json.dumps({"event":"ready"})+"\n");sys.stdout.flush()
    while True:
        try:
            line=sys.stdin.readline()
            if not line:break
            req=json.loads(line.strip());cmd=req.get("cmd")
            if cmd=="update_candidates":
                with candidates_lock:candidates=req.get("candidates",[])
                log(f"Updated candidates list: {len(candidates)} persons")
                sys.stdout.write(json.dumps({"status":"success","cmd":"update_candidates"})+"\n");sys.stdout.flush()
            elif cmd=="start_stream":
                camera_id=req["camera_id"]
                with streams_lock:
                    if camera_id in stream_stop_events:stream_stop_events[camera_id].set()
                    stop_event=threading.Event();stream_stop_events[camera_id]=stop_event
                    args=(camera_id,req.get("camera_name",camera_id),req["rtsp_url"],req.get("threshold",0.50),req.get("dis_type",0),req["crops_dir"],req.get("line_crossing_enabled",False),req.get("line_y",0.6),req.get("line_direction","in"),req.get("line_x_start",0.0),req.get("line_x_end",1.0),stop_event)
                    t=threading.Thread(target=rtsp_stream_processor,args=args,daemon=True);stream_threads[camera_id]=t;t.start()
                sys.stdout.write(json.dumps({"status":"success","cmd":"start_stream","camera_id":camera_id})+"\n");sys.stdout.flush()
            elif cmd=="stop_stream":
                camera_id=req["camera_id"]
                with streams_lock:
                    if camera_id in stream_stop_events:stream_stop_events[camera_id].set()
                sys.stdout.write(json.dumps({"status":"success","cmd":"stop_stream","camera_id":camera_id})+"\n");sys.stdout.flush()
            elif cmd=="process_image":
                results=process_single_image(req["image_path"],req.get("threshold",0.50),req.get("dis_type",0),req["crops_dir"])
                sys.stdout.write(json.dumps({"status":"success","request_id":req.get("request_id"),"faces":results})+"\n");sys.stdout.flush()
            elif cmd=="extract_embedding":
                emb=extract_best_face_embedding(req["image_path"],req.get("enforce_quality",True));sys.stdout.write(json.dumps({"status":"success","request_id":req.get("request_id"),"embedding":emb})+"\n");sys.stdout.flush()
            elif cmd=="process_enroll_video":
                faces=process_video_for_enrollment(req["video_path"],req["crops_dir"],req.get("person_id"));sys.stdout.write(json.dumps({"status":"success","request_id":req.get("request_id"),"faces":faces})+"\n");sys.stdout.flush()
            elif cmd=="update_settings":
                sys.stdout.write(json.dumps({"status":"success","cmd":"update_settings"})+"\n");sys.stdout.flush()
            else:
                sys.stdout.write(json.dumps({"status":"error","message":f"Unknown command: {cmd}"})+"\n");sys.stdout.flush()
        except Exception as e:
            log(f"Main loop error: {str(e)}");sys.stdout.write(json.dumps({"status":"error","message":str(e)})+"\n");sys.stdout.flush()

if __name__=="__main__":main()
