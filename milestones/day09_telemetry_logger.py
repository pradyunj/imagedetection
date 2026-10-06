# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 9: Data Logging (The Researcher's Ledger - CSV Telemetry Stream)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Phase 1: Open CSV in append mode and write header
csv_filename = "flight_data.csv"
csv_file = open(csv_filename, "a", newline="")
csv_writer = csv.writer(csv_file)

# Write header row if file is empty
if csv_file.tell() == 0:
    csv_writer.writerow(["Timestamp", "Object", "X_Coord", "Y_Coord"])
    csv_file.flush()

print(f"[SYSTEM] Logging telemetry to {csv_filename}...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model.predict(source=frame, verbose=False)

    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        if conf > 0.45:
            cls_name = model.names[int(box.cls[0].item())]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Phase 2: Stream real-time coordinates to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            csv_writer.writerow([timestamp, cls_name, cx, cy])
            csv_file.flush()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} ({cx}, {cy})", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break

    cv2.imshow("Pradyun's AI - Real-time Telemetry Logger", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Phase 3: Clean shutdown and safe file closure
cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"[SYSTEM] Day 9 Complete! Telemetry logged and {csv_filename} saved safely.")
