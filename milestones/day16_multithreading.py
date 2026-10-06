# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 16: Producer-Consumer Multi-Threading Pipeline (Decoupled I/O & Inference)
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import threading
import queue
import time
from ultralytics import YOLO

class MultiThreadedVisionSystem:
    def __init__(self, source=0):
        self.source = source
        self.frame_queue = queue.Queue(maxsize=5)
        self.model = YOLO("yolov8n.pt")
        self.running = True

    # Producer Thread: Continuously grabs hardware frames
    def producer_worker(self, cap):
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        self.frame_queue.put(None)

    # Consumer Thread: Performs inference and renders output
    def start(self):
        cap = cv2.VideoCapture(self.source)
        t = threading.Thread(target=self.producer_worker, args=(cap,))
        t.daemon = True
        t.start()

        print("[SYSTEM] Multi-threaded Producer-Consumer pipeline active.")

        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break

            results = self.model.predict(source=frame, verbose=False)
            cv2.putText(frame, "Decoupled Multi-Threaded Pipeline", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Pradyun's AI - Multi-Threaded Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()
        t.join()
        print("[SYSTEM] Day 16 Complete! Producer-Consumer threads synchronized.")

if __name__ == "__main__":
    system = MultiThreadedVisionSystem(source=0)
    system.start()
