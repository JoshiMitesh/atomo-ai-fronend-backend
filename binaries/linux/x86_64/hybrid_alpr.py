#!/usr/bin/env python3
import ultimateAlprSdk
import cv2
import json
import numpy as np
import os
import sys
from paddleocr import PaddleOCR

# ====================== CONFIG ======================
VIDEO_PATH = 0
ASSETS_PATH = os.path.abspath("../../../assets")

JSON_CONFIG = {
    "debug_level": "info",
    "gpgpu_enabled": True,
    "assets_folder": ASSETS_PATH,
    "detect_minscore": 0.15,
    "pyramidal_search_enabled": True,
    "pyramidal_search_sensitivity": 0.4,
    "recogn_minscore": 0.1,
    "recogn_rectify_enabled": True,
    "charset": "latin"
}
# ====================================================

def check_result(operation, result):
    if not result.isOK():
        print(f"[ERROR] {operation}: {result.phrase()}")
        sys.exit(-1)
    print(f"[OK] {operation}")

def main():
    # Initialize ultimateALPR
    check_result("Init ultimateALPR", ultimateAlprSdk.UltAlprSdkEngine_init(json.dumps(JSON_CONFIG)))

    # Initialize PaddleOCR (fixed for new version)
    print("Loading PaddleOCR... (first time may take time)")
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en'
    )
    print("PaddleOCR loaded.\n")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video")
        sys.exit(-1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    frame_count = 0
    last_text = ""

    print("Running Hybrid ALPR... Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        height, width = frame.shape[:2]

        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        # 1. Detect plate with ultimateALPR
        result = ultimateAlprSdk.UltAlprSdkEngine_process(
            ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_BGR24,
            frame.tobytes(),
            width,
            height
        )

        display_frame = frame.copy()

        if result.isOK() and result.numPlates() > 0:
            try:
                data = json.loads(result.json())
                for plate in data.get("plates", []):
                    box = plate.get("warpedBox")
                    if not box or len(box) < 8:
                        continue

                    pts = np.array([[int(box[i]), int(box[i+1])] for i in range(0, 8, 2)], np.int32)

                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]
                    x1, x2 = max(0, min(x_coords)), min(width, max(x_coords))
                    y1, y2 = max(0, min(y_coords)), min(height, max(y_coords))

                    if x2 - x1 < 20 or y2 - y1 < 10:
                        continue

                    # Crop the plate
                    plate_crop = frame[y1:y2, x1:x2]

                    # 2. Run PaddleOCR on the crop
                    ocr_result = ocr.ocr(plate_crop, cls=True)

                    text = ""
                    if ocr_result and ocr_result[0]:
                        texts = [line[1][0] for line in ocr_result[0]]
                        text = "".join(texts).upper().replace(" ", "")

                    if text and text != last_text and len(text) >= 6:
                        print(f"Frame {frame_count:5d} | {text}")
                        last_text = text

                    # Draw
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    cv2.putText(display_frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                print("Error:", e)

        cv2.imshow("Hybrid ALPR (ultimateALPR + PaddleOCR)", display_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ultimateAlprSdk.UltAlprSdkEngine_deInit()
    print("Finished.")

if __name__ == "__main__":
    main()
