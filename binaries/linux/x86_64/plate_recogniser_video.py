#!/usr/bin/env python3
import cv2
import requests
import time
import threading
from queue import Queue

# ====================== CONFIG ======================
API_TOKEN = "ae87b1d9cbf46dba0fb33ec5a24f147d2152fcfd"          # ← Put your real token here
VIDEO_PATH = "numberplate.mp4"
REGIONS = ["in"]                           # India
PROCESS_EVERY_N_FRAMES = 8                 # Higher = less API calls (safer for free plan)
# ====================================================

result_queue = Queue()
latest_result = None

def api_worker():
    """Background thread for API calls"""
    global latest_result
    while True:
        frame = result_queue.get()
        if frame is None:
            break

        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        try:
            response = requests.post(
                "https://api.platerecognizer.com/v1/plate-reader/",
                files=dict(upload=img_encoded.tobytes()),
                data=dict(regions=REGIONS),
                headers={"Authorization": f"Token {API_TOKEN}"},
                timeout=10
            )
            if response.status_code in [200, 201]:
                latest_result = response.json()
        except Exception as e:
            print("API Error:", e)

        result_queue.task_done()

def main():
    if API_TOKEN == "YOUR_API_TOKEN_HERE":
        print("Please put your API token first!")
        return

    # Start background API thread
    worker = threading.Thread(target=api_worker, daemon=True)
    worker.start()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = max(1, int(1000 / fps)) if fps > 0 else 30
    frame_count = 0
    last_plates = set()

    print("Running... Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        # Send frame to API every N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            if result_queue.empty():
                result_queue.put(frame.copy())

        # Draw latest result
        if latest_result and "results" in latest_result:
            for plate_info in latest_result["results"]:
                plate = plate_info.get("plate", "").upper()
                score = plate_info.get("score", 0) * 100
                box = plate_info.get("box", {})

                if plate and plate not in last_plates:
                    print(f"Frame {frame_count:5d} | {plate:12s} | {score:5.1f}%")
                    last_plates.add(plate)

                if box:
                    x1 = int(box.get("xmin", 0))
                    y1 = int(box.get("ymin", 0))
                    x2 = int(box.get("xmax", 0))
                    y2 = int(box.get("ymax", 0))
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{plate}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Plate Recognizer", display_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Cleanup
    result_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    print("Finished.")

if __name__ == "__main__":
    main()
