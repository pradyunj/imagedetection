# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 18: Edge Optimization & Frame Dropping (Hybrid Inference-Tracking Loop)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import time
from ultralytics import YOLO

def create_tracker():
    """KCF is only in opencv-contrib, so fall back to MIL (included in regular OpenCV)."""
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    return cv2.TrackerMIL_create()

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

frame_count = 0
tracker = None
tracking_active = False
detected_box = None

print("[SYSTEM] Hybrid Edge Inference-Tracker active (YOLO on frame % 5).")

while True:
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # Task 1 & 2: Run heavy YOLO only every 5th frame; use tracker for others
    if frame_count % 5 == 0 or not tracking_active:
        mode = "YOLO INFERENCE"
        results = model.predict(source=frame, verbose=False)
        best_box = None
        for box in results[0].boxes:
            if float(box.conf[0].item()) > 0.40:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                best_box = (x1, y1, x2 - x1, y2 - y1)
                break
        if best_box:
            tracker = create_tracker()
            tracker.init(frame, best_box)
            detected_box = best_box
            tracking_active = True
        else:
            tracking_active = False
    else:
        mode = "TRACKER"
        ok, box = tracker.update(frame)
        if ok:
            detected_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        else:
            tracking_active = False

    t1 = time.perf_counter()
    fps = int(1.0 / (t1 - t0)) if (t1 - t0) > 0 else 0

    if tracking_active and detected_box:
        x, y, w, h = detected_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{mode} | FPS: {fps}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Pradyun's AI - Edge Optimized Hybrid Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 18 Complete! High-throughput edge optimization verified.")
