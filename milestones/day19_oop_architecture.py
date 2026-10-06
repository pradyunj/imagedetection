# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 19: Object-Oriented Refactoring (RaptorDefenseSystem Class)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import os
from ultralytics import YOLO

class RaptorDefenseSystem:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.model = YOLO("yolov8n.pt")
        print("[SYSTEM] RaptorDefenseSystem initialized.")

    def load_model(self):
        return self.model

    def check_collision(self, cx, cy):
        # Turbine danger zone bounding test
        return (200 <= cx <= 440 and 100 <= cy <= 380)

    def start(self):
        cap = cv2.VideoCapture(self.video_source)
        print("[SYSTEM] System loop started cleanly via OOP interface.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, "RaptorDefenseSystem: Object-Oriented Architecture", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Pradyun's AI - OOP Architecture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = RaptorDefenseSystem(video_source=0)
    system.start()
