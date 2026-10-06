# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 6: Targeted Object Filtering (The "Sniper" Focus)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
from ultralytics import YOLO

print("[SYSTEM] Initializing Targeted Sniper Filter...")
model = YOLO("yolov8n.pt")

# Target: Bird (COCO class ID 14), Person (0), or Cell Phone (67)
TARGET_CLASS_IDS = [0, 14, 67]
CONFIDENCE_THRESHOLD = 0.60

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not available. Exiting.")
    exit()

print(f"[SYSTEM] Monitoring for Target IDs: {TARGET_CLASS_IDS} with Conf > {CONFIDENCE_THRESHOLD:.0%}")
print("[SYSTEM] Press 'q' to quit.")

alert_logged = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, verbose=False)
    target_in_frame = False
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        
        # Phase 1 & 2: ID Logic Gate + Confidence Thresholding
        if cls_id in TARGET_CLASS_IDS and conf >= CONFIDENCE_THRESHOLD:
            target_in_frame = True
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            
            # Phase 3: Console Alert Trigger on Acquisition
            if not alert_logged:
                print(f"[ALERT] TARGET ACQUIRED! {cls_name.upper()} (Confidence: {conf:.1%})")
                alert_logged = True
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"[LOCKED] {cls_name}: {conf:.0%}"
            cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not target_in_frame:
        alert_logged = False

    cv2.imshow("Pradyun's AI - Targeted Sniper Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 6 Complete! Target filtering active.")
