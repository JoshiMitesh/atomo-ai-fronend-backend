import cv2
import json
import os
import sys
import time

# ============================================================
# PATHS
# ============================================================

SDK_ROOT = os.path.expanduser("~/ultimateALPR-SDK")
BIN_DIR = os.path.join(SDK_ROOT, "binaries", "linux", "x86_64")
ASSETS_DIR = os.path.join(SDK_ROOT, "assets")

sys.path.insert(0, BIN_DIR)

import ultimateAlprSdk


# ============================================================
# CONFIGURATION
# ============================================================

config = {
    "debug_level": "error",

    "assets_folder": ASSETS_DIR,
    "charset": "latin",

    # CPU
    "num_threads": 4,
    "max_jobs": 1,

    # OpenVINO CPU
    "openvino_enabled": True,
    "openvino_device": "CPU",

    # No CUDA/NPU/TensorRT
    "gpgpu_enabled": False,
    "npu_enabled": False,
    "trt_enabled": False,

    # Detection
    "detect_minscore": 0.1,
    "car_noplate_detect_min_score": 0.8,

    # Pyramidal search
    "pyramidal_search_enabled": True,
    "pyramidal_search_sensitivity": 1.0,
    "pyramidal_search_minscore": 0.3,
    "pyramidal_search_min_image_size_inpixels": 800,

    # Recognition
    "recogn_minscore": 0.3,
    "recogn_rectify_enabled": True,
    "recogn_score_type": "min",

    # Disable additional classifiers
    "klass_lpci_enabled": False,
    "klass_vcr_enabled": False,
    "klass_vmmr_enabled": False,
    "klass_vbsr_enabled": False,

    # Image enhancement
    "ienv_enabled": False,
}


# ============================================================
# INITIALIZE ALPR
# ============================================================

print("==============================================")
print(" UltimateALPR Live Webcam")
print("==============================================")
print("SDK       :", SDK_ROOT)
print("Assets    :", ASSETS_DIR)
print("OpenVINO  : CPU")
print("Threads   :", config["num_threads"])
print()

result = ultimateAlprSdk.UltAlprSdkEngine_init(
    json.dumps(config)
)

if not result.isOK():
    print("ERROR: ALPR initialization failed")
    print(result.phrase())
    sys.exit(1)

print("ALPR SDK initialized successfully")

# Warm up the engine so the first live detection isn't unusually slow.
#warmup = ultimateAlprSdk.UltAlprSdkEngine_warmUp()
warmup = ultimateAlprSdk.UltAlprSdkEngine_warmUp("")
print("Warm-up:", warmup.isOK(), warmup.phrase())

if not warmup.isOK():
    print("WARNING: warmUp failed:")
    print(warmup.phrase())
else:
    print("ALPR warm-up complete")


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")

    ultimateAlprSdk.UltAlprSdkEngine_deInit()

    sys.exit(1)


# Request 1280x720.
# The camera may choose another supported resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(
    f"Webcam resolution: "
    f"{actual_width}x{actual_height}"
)

print()
print("Press Q or ESC to quit")
print()


# ============================================================
# PERFORMANCE
# ============================================================

# Your static test was around 292 ms.
#
# Running ALPR on every webcam frame would make the display
# choppy. Instead:
#
# webcam/display: ~30 FPS
# ALPR:            every 5th frame
#
ALPR_INTERVAL = 5

frame_number = 0

last_result = None
last_alpr_ms = 0.0

fps = 0.0
fps_counter = 0
fps_start = time.perf_counter()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            print("ERROR: Failed to read webcam frame")
            break

        frame_number += 1
        fps_counter += 1

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed = time.perf_counter() - fps_start

        if elapsed >= 1.0:

            fps = fps_counter / elapsed

            fps_counter = 0
            fps_start = time.perf_counter()


        # ----------------------------------------------------
        # ALPR
        # ----------------------------------------------------

        if frame_number % ALPR_INTERVAL == 0:

            # OpenCV:
            # BGR, uint8
            #
            # UltimateALPR:
            # RGB24

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            height, width = rgb.shape[:2]

            stride = width * 3

            start = time.perf_counter()

            result = ultimateAlprSdk.UltAlprSdkEngine_process(
                ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGB24,
                rgb.tobytes(),
                width,
                height,
                stride,
                1
            )

            last_alpr_ms = (
                time.perf_counter() - start
            ) * 1000.0

            if result.isOK():

                try:
                    last_result = json.loads(
                        result.json()
                    )

                except Exception as e:

                    print(
                        "JSON parsing error:",
                        e
                    )

                    last_result = None

            else:

                print(
                    "ALPR processing error:",
                    result.phrase()
                )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display = frame.copy()

        plates = []

        if last_result:

            plates = last_result.get(
                "plates",
                []
            )


        # ----------------------------------------------------
        # DRAW DETECTED PLATES
        # ----------------------------------------------------

        for plate in plates:

            text = plate.get(
                "text",
                ""
            )

            # Character confidence values
            confidences = plate.get(
                "confidences",
                []
            )

            if confidences:

                confidence = (
                    sum(confidences)
                    / len(confidences)
                )

            else:

                confidence = 0.0


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            box = plate.get(
                "warpedBox",
                []
            )

            points = []

            if len(box) >= 8:

                for i in range(0, 8, 2):

                    points.append(
                        (
                            int(box[i]),
                            int(box[i + 1])
                        )
                    )


            if len(points) == 4:

                # Draw plate polygon

                for i in range(4):

                    cv2.line(
                        display,
                        points[i],
                        points[(i + 1) % 4],
                        (0, 255, 0),
                        2
                    )

                x = min(
                    p[0] for p in points
                )

                y = min(
                    p[1] for p in points
                )

            else:

                x = 20
                y = 140


            # ------------------------------------------------
            # Plate label
            # ------------------------------------------------

            label = (
                f"{text} "
                f"{confidence:.1f}%"
            )

            label_width = max(
                250,
                len(label) * 12
            )

            label_y = max(
                35,
                y
            )

            cv2.rectangle(
                display,
                (
                    x,
                    label_y - 35
                ),
                (
                    x + label_width,
                    label_y
                ),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                display,
                label,
                (
                    x + 5,
                    label_y - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )


        # ====================================================
        # STATUS
        # ====================================================

        cv2.putText(
            display,
            f"Camera FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"ALPR: {last_alpr_ms:.0f} ms",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Plates: {len(plates)}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if plates else (0, 0, 255),
            2
        )

        cv2.putText(
            display,
            "UltimateALPR",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "UltimateALPR - Live Webcam",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break


finally:

    print()
    print("Stopping webcam...")

    cap.release()

    cv2.destroyAllWindows()

    print("Deinitializing UltimateALPR...")

    result = (
        ultimateAlprSdk.UltAlprSdkEngine_deInit()
    )

    if result.isOK():

        print("ALPR SDK deinitialized successfully")

    else:

        print(
            "SDK deinitialization error:",
            result.phrase()
        )

    print("Done.")
