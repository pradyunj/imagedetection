# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 4: YOLO AI Integration (Static Image Model Inference Pipeline)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
import os
from ultralytics import YOLO

print("[SYSTEM] Loading pre-trained YOLOv8 nano model...")
model = YOLO("yolov8n.pt")

# Phase 2: Load or create test image
test_image_path = "test_image.jpg"
if not os.path.exists(test_image_path):
    print("[SYSTEM] Generating synthetic test canvas with geometric targets...")
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (100, 150), (250, 350), (40, 40, 40), -1)
    cv2.circle(img, (450, 250), 60, (0, 100, 200), -1)
    cv2.putText(img, "YOLO Test Image Target", (150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(test_image_path, img)

print(f"[SYSTEM] Running model inference on: {test_image_path}...")
results = model.predict(source=test_image_path, save=True, verbose=False)

# Phase 3: Inspecting the AI's Brain - Parse predictions
print("\n--- Inference Results ---")
boxes = results[0].boxes
print(f"Total objects detected: {len(boxes)}")

for i, box in enumerate(boxes):
    cls_id = int(box.cls[0].item())
    conf = float(box.conf[0].item())
    cls_name = model.names[cls_id]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    
    print(f"Object #{i+1}:")
    print(f"  Class: {cls_name} (ID: {cls_id})")
    print(f"  Confidence: {conf:.2%}")
    print(f"  Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

print("\n[SYSTEM] Day 4 Complete! Inference results printed successfully.")
