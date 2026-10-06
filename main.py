# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Location: Altamont Pass Wind Resource Area, California
# Architecture: End-to-End Autonomous Collision Prevention Command Center
# ==============================================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import threading
import queue
from datetime import datetime
from ultralytics import YOLO

class RaptorDefenseSystem:
    def __init__(self, video_source="birdtest.mp4"):
        # Auto-detect video source: fallback to webcam index 0 if video asset not present
        if isinstance(video_source, str) and not os.path.exists(video_source):
            print(f"[SYSTEM] Notice: Video asset '{video_source}' not found. Defaulting to live webcam (Index 0).")
            self.video_source = 0
        else:
            self.video_source = video_source

        # Load weights: use custom trained best.pt if present, otherwise yolov8n.pt
        model_weights = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
        print(f"[SYSTEM] Initializing neural network weights from: {model_weights}")
        self.model = YOLO(model_weights)

        # Threading queue & output storage
        self.frame_queue = queue.Queue(maxsize=5)
        self.recordings_dir = os.path.join('recordings')
        os.makedirs(self.recordings_dir, exist_ok=True)

        # Turbine danger airspace coordinates (Airspace Hazard Zone)
        # Bounding box: X in [200, 440], Y in [100, 380]
        self.zone_pts = np.array([[200, 100], [440, 100], [440, 380], [200, 380]], np.int32).reshape((-1, 1, 2))
        self.rect_zone = (200, 100, 240, 280)

    def read_frames(self, cap):
        """Producer Thread: Continuously grabs hardware/video frames."""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Drop older frames if buffer is full to prevent latency buildup
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        self.frame_queue.put(None)

    def write_telemetry(self, csv_writer, name, cx, cy):
        """Researcher's Ledger: Streams real-time detection telemetry to CSV."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        csv_writer.writerow([current_time, name, cx, cy])

    def check_collision(self, cx, cy, trail, frame):
        """Spatial Intersection & Predictive Trajectory Vector Collision Engine."""
        alert = False
        # Direct spatial breach check
        if 200 <= cx <= 440 and 100 <= cy <= 380:
            alert = True

        # Day 17: Velocity Vector & Forward Projection (predict 30 frames ahead)
        if len(trail) >= 5:
            px, py = trail[-5]
            vx = cx - px
            vy = cy - py
            fx = int(cx + (vx * 6))
            fy = int(cy + (vy * 6))

            predictive_hit, _, _ = cv2.clipLine(self.rect_zone, (cx, cy), (fx, fy))
            if predictive_hit:
                alert = True
                cv2.line(frame, (cx, cy), (fx, fy), (0, 165, 255), 3)  # Orange warning vector
            else:
                cv2.line(frame, (cx, cy), (fx, fy), (0, 255, 0), 2)    # Green safe vector
        return alert, frame

    def render_dashboard(self, frame, fps_val, cx, cy, conf, tracking_mode, alert_triggered):
        """Day 20: Industrial Mission Control Command Center HUD."""
        sidebar_w = 260
        sidebar = np.zeros((480, sidebar_w, 3), dtype=np.uint8)

        # Header Title
        cv2.putText(sidebar, "TURBINE DEFENSE", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(sidebar, "COMMAND CENTER", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.line(sidebar, (20, 85), (240, 85), (70, 70, 70), 1)

        # Telemetry Block
        cv2.putText(sidebar, "SYSTEM TELEMETRY", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(sidebar, f"FPS: {fps_val}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(sidebar, f"ENGINE: {tracking_mode}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        status_text = "ACQUIRED" if cx is not None else "SCANNING..."
        status_color = (0, 255, 0) if cx is not None else (100, 100, 100)
        cv2.putText(sidebar, f"TARGET: {status_text}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

        coord_text = f"X: {cx}  Y: {cy}" if cx is not None else "X: ---  Y: ---"
        cv2.putText(sidebar, coord_text, (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        conf_text = f"CONF: {conf:.2f}" if conf > 0 else "CONF: N/A"
        cv2.putText(sidebar, conf_text, (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.line(sidebar, (20, 290), (240, 290), (70, 70, 70), 1)

        # Big Industrial Indicator Button
        btn_center = (130, 380)
        btn_radius = 48

        if alert_triggered:
            flash_on = (int(time.time() * 6) % 2 == 0)
            btn_color = (0, 0, 255) if flash_on else (0, 0, 100)
            cv2.circle(sidebar, btn_center, btn_radius, btn_color, -1)
            cv2.circle(sidebar, btn_center, btn_radius + 4, (0, 0, 255), 2)
            cv2.putText(sidebar, "BRAKE", (103, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(sidebar, "ENGAGED", (90, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.circle(sidebar, btn_center, btn_radius, (0, 180, 0), -1)
            cv2.circle(sidebar, btn_center, btn_radius + 4, (0, 255, 0), 2)
            cv2.putText(sidebar, "AIRSPACE", (90, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(sidebar, "SECURE", (97, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return np.hstack([frame, sidebar])

    def visualize_data(self):
        """Day 13: Matplotlib Post-Flight Trajectory Analysis."""
        list_of_files = glob.glob(os.path.join(self.recordings_dir, '*.csv'))
        if not list_of_files:
            return
        latest_file = max(list_of_files, key=os.path.getctime)
        try:
            df = pd.read_csv(latest_file)
            if df.empty or 'X_Coord' not in df.columns:
                return

            plt.figure(figsize=(10, 6))
            plt.plot(df['X_Coord'], df['Y_Coord'], marker='o', linestyle='-', color='red', markersize=4, label='Raptor Trajectory')
            plt.gca().add_patch(plt.Rectangle((200, 100), 240, 280, color='blue', alpha=0.2, label='Turbine Hazard Zone'))
            plt.title('Altamont Pass Raptor Defense: Flight Trajectory Telemetry')
            plt.xlabel('Camera X Coordinate (Pixels)')
            plt.ylabel('Camera Y Coordinate (Pixels)')
            plt.xlim(0, 640)
            plt.ylim(0, 480)
            plt.gca().invert_yaxis()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(self.recordings_dir, 'latest_trajectory_plot.png')
            plt.savefig(plot_path)
            print(f"[SYSTEM] Flight trajectory visualization saved to: {plot_path}")
            plt.show()
        except Exception as e:
            print(f"[SYSTEM] Visualization notice: {e}")

    def start(self):
        """Execute the full 20-Day Integrated Pipeline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("[ERROR] Could not open video source.")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 30
        dashboard_w = 640 + 260
        video_out_path = os.path.join(self.recordings_dir, f'{timestamp}_video_processing.mp4')
        out = cv2.VideoWriter(
            video_out_path,
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (dashboard_w, 480)
        )

        csv_path = os.path.join(self.recordings_dir, f'{timestamp}_telemetry.csv')
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Object', 'X_Coord', 'Y_Coord'])

        # Start Producer Thread
        producer_thread = threading.Thread(target=self.read_frames, args=(cap,))
        producer_thread.daemon = True
        producer_thread.start()

        trail = []
        frame_count = 0
        tracker = None
        tracking_active = False
        fps_display = 0

        print("[SYSTEM] Command Center online! Press 'q' to shut down safely.")

        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break

            start_time = time.time()
            frame = cv2.resize(frame, (640, 480))
            frame_count += 1

            zone_color = (0, 0, 255)
            alert_triggered = False
            detected_box = None
            conf_score = 0.0
            tracking_mode = "YOLO INFERENCE"

            # Day 18: Edge Optimization (YOLO every 5 frames; KCF Tracking in-between)
            if frame_count % 5 == 0 or not tracking_active:
                results = self.model.predict(source=frame, verbose=False)
                best_conf = 0

                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = self.model.names.get(cls_id, "").lower()

                    # Target raptors (Bird COCO ID 14, Eagle, or general foreground detection)
                    if (cls_id in [0, 14] or "bird" in cls_name or "eagle" in cls_name) and conf > 0.35:
                        if conf > best_conf:
                            best_conf = conf
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detected_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                            conf_score = conf

                if detected_box:
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, detected_box)
                    tracking_active = True
                else:
                    tracking_active = False
            else:
                tracking_mode = "KCF TRACKING"
                success, box = tracker.update(frame)
                if success:
                    detected_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                else:
                    tracking_active = False

            cx, cy = None, None
            if detected_box:
                x, y, w, h = detected_box
                cx = int(x + w / 2)
                cy = int(y + h / 2)

                trail.append((cx, cy))
                if len(trail) > 30:
                    trail.pop(0)

                alert_triggered, frame = self.check_collision(cx, cy, trail, frame)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Target {conf_score:.2f}" if tracking_mode == "YOLO INFERENCE" else "Target (Tracked)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                self.write_telemetry(csv_writer, "Raptor", cx, cy)

            if alert_triggered:
                print("[CRITICAL] TURBINE BRAKE ENGAGED - PREDICTIVE OR DIRECT BREACH")
                if int(time.time() * 8) % 2 == 0:
                    zone_color = (0, 255, 255)

            # Draw turbine danger zone polygon with alpha blending
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.zone_pts], zone_color)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.polylines(frame, [self.zone_pts], True, zone_color, 2)

            # Draw historical motion trail
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 2)

            end_time = time.time()
            loop_time = end_time - start_time
            if loop_time > 0:
                fps_display = int(1 / loop_time)

            dashboard_frame = self.render_dashboard(
                frame, fps_display, cx, cy, conf_score, tracking_mode, alert_triggered
            )

            out.write(dashboard_frame)
            cv2.imshow("Raptor Defense System - Command Center", dashboard_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        csv_file.close()
        cv2.destroyAllWindows()
        producer_thread.join()

        print("[SYSTEM] Clean shutdown complete. Generating flight trajectory analytics...")
        self.visualize_data()

if __name__ == "__main__":
    import sys
    # Default to birdtest.mp4 if present; otherwise webcam 0
    default_source = "birdtest.mp4" if os.path.exists("birdtest.mp4") else 0
    video_arg = sys.argv[1] if len(sys.argv) > 1 else default_source

    if video_arg in ["0", "webcam", "camera"]:
        video_arg = 0

    system = RaptorDefenseSystem(video_source=video_arg)
    system.start()
