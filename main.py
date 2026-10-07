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

def create_tracker():
    """KCF is only in opencv-contrib, so fall back to MIL (included in standard OpenCV)."""
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    return cv2.TrackerMIL_create()

class RaptorDefenseSystem:
    def __init__(self, video_source="eagle_flying.mp4"):
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
        self.frame_queue = queue.Queue(maxsize=10)
        self.running = True
        self.recordings_dir = os.path.join('recordings')
        os.makedirs(self.recordings_dir, exist_ok=True)

        # Turbine danger airspace coordinates (Airspace Hazard Zone)
        # Bounding box: X in [200, 440], Y in [100, 380]
        self.zone_pts = np.array([[200, 100], [440, 100], [440, 380], [200, 380]], np.int32).reshape((-1, 1, 2))
        self.rect_zone = (200, 100, 240, 280)

        # State filters for stability
        self.smooth_x = None
        self.smooth_y = None
        self.brake_latch_until = 0.0  # Hysteresis safety hold timer

    def read_frames(self, cap):
        """Producer Thread: Continuously grabs hardware/video frames."""
        is_webcam = isinstance(self.video_source, int)
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if is_webcam:
                # Webcam: keep feed live by discarding backlogged frames
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            else:
                # Video file: block until consumer is ready so no frames are lost
                while self.running:
                    try:
                        self.frame_queue.put(frame, timeout=0.2)
                        break
                    except queue.Full:
                        pass
        try:
            self.frame_queue.put(None, timeout=0.2)
        except queue.Full:
            pass

    def write_telemetry(self, csv_writer, name, cx, cy):
        """Researcher's Ledger: Streams real-time detection telemetry to CSV."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        csv_writer.writerow([current_time, name, cx, cy])

    def check_collision(self, scx, scy, trail, frame):
        """Spatial Intersection & Predictive Velocity Vector Collision Engine."""
        instant_alert = False

        # 1. Direct spatial breach check (is the bird currently inside the rotor box?)
        if 200 <= scx <= 440 and 100 <= scy <= 380:
            instant_alert = True

        # 2. Predictive Trajectory Vector (using smoothed history over past 8 frames)
        if len(trail) >= 8:
            # Measure velocity over a stable baseline to eliminate frame jitter
            dx = trail[-1][0] - trail[-8][0]
            dy = trail[-1][1] - trail[-8][1]
            vx = dx / 7.0
            vy = dy / 7.0

            # Project path 18 frames forward (~0.6 seconds in advance)
            fx = int(scx + vx * 18)
            fy = int(scy + vy * 18)

            predictive_hit, _, _ = cv2.clipLine(self.rect_zone, (scx, scy), (fx, fy))
            if predictive_hit:
                instant_alert = True
                cv2.line(frame, (scx, scy), (fx, fy), (0, 140, 255), 3)  # Orange warning vector
                cv2.circle(frame, (fx, fy), 4, (0, 140, 255), -1)
            else:
                cv2.line(frame, (scx, scy), (fx, fy), (0, 255, 0), 2)    # Green safe vector

        # 3. Industrial Latch / Hysteresis: Hold brake engaged for at least 1.8 seconds
        # This prevents mechanical brake chatter (rapid on/off flickering)
        if instant_alert:
            self.brake_latch_until = time.time() + 1.8

        alert_active = (time.time() < self.brake_latch_until)
        return alert_active, frame

    def render_dashboard(self, frame, fps_val, cx, cy, conf, tracking_mode, alert_triggered):
        """Day 20: Industrial Mission Control Command Center HUD."""
        sidebar_w = 260
        sidebar = np.zeros((480, sidebar_w, 3), dtype=np.uint8)

        # Header Title
        cv2.putText(sidebar, "ALTAMONT PASS", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(sidebar, "TURBINE MONITOR", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.line(sidebar, (20, 85), (240, 85), (70, 70, 70), 1)

        # Telemetry Block
        cv2.putText(sidebar, "SYSTEM STATUS", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(sidebar, f"FPS: {fps_val}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(sidebar, f"MODE: {tracking_mode}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        status_text = "DETECTED" if cx is not None else "SCANNING..."
        status_color = (0, 255, 0) if cx is not None else (100, 100, 100)
        cv2.putText(sidebar, f"BIRD: {status_text}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

        coord_text = f"X: {cx}  Y: {cy}" if cx is not None else "X: ---  Y: ---"
        cv2.putText(sidebar, coord_text, (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        conf_text = f"CONF: {conf:.2f}" if conf > 0 else "CONF: N/A"
        cv2.putText(sidebar, conf_text, (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.line(sidebar, (20, 290), (240, 290), (70, 70, 70), 1)

        # Big Industrial Indicator Button
        btn_center = (130, 380)
        btn_radius = 48

        if alert_triggered:
            # Steady active red light with subtle pulsing
            pulse = int(180 + 75 * np.sin(time.time() * 8))
            cv2.circle(sidebar, btn_center, btn_radius, (0, 0, pulse), -1)
            cv2.circle(sidebar, btn_center, btn_radius + 4, (0, 0, 255), 2)
            cv2.putText(sidebar, "BRAKE", (103, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(sidebar, "ENGAGED", (90, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.circle(sidebar, btn_center, btn_radius, (0, 180, 0), -1)
            cv2.circle(sidebar, btn_center, btn_radius + 4, (0, 255, 0), 2)
            cv2.putText(sidebar, "ALL", (90, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(sidebar, "CLEAR", (97, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return np.hstack([frame, sidebar])

    def visualize_data(self):
        """Day 13: Matplotlib Post-Flight Trajectory Analysis with Noise Filtering."""
        list_of_files = glob.glob(os.path.join(self.recordings_dir, '*.csv'))
        if not list_of_files:
            return
        latest_file = max(list_of_files, key=os.path.getctime)
        try:
            df = pd.read_csv(latest_file)
            if df.empty or 'X_Coord' not in df.columns:
                return

            # Apply rolling filter so the plot shows clean aerodynamic flight path instead of jitter
            df['X_Smooth'] = df['X_Coord'].rolling(window=7, min_periods=1, center=True).mean()
            df['Y_Smooth'] = df['Y_Coord'].rolling(window=7, min_periods=1, center=True).mean()

            # Downsample slightly for clean scatter points
            sample_df = df.iloc[::4]

            plt.figure(figsize=(10, 6))
            # Smooth trajectory curve
            plt.plot(df['X_Smooth'], df['Y_Smooth'], linestyle='-', color='crimson', linewidth=2.5, label='Raptor Trajectory')
            plt.scatter(sample_df['X_Smooth'], sample_df['Y_Smooth'], color='darkred', s=20, zorder=3)

            # Hazard Zone
            plt.gca().add_patch(plt.Rectangle((200, 100), 240, 280, color='royalblue', alpha=0.25, label='Turbine Hazard Zone'))

            plt.title('Altamont Pass Raptor Defense: Flight Trajectory Telemetry', fontsize=13, fontweight='bold')
            plt.xlabel('Camera X Coordinate (Pixels)', fontsize=11)
            plt.ylabel('Camera Y Coordinate (Pixels)', fontsize=11)
            plt.xlim(0, 640)
            plt.ylim(0, 480)
            plt.gca().invert_yaxis()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='upper right')
            plt.tight_layout()

            plot_path = os.path.join(self.recordings_dir, 'latest_trajectory_plot.png')
            plt.savefig(plot_path, dpi=180)
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

        is_webcam = isinstance(self.video_source, int)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0
        frame_interval = 1.0 / video_fps

        dashboard_w = 640 + 260
        video_out_path = os.path.join(self.recordings_dir, f'{timestamp}_video_processing.mp4')
        out = cv2.VideoWriter(
            video_out_path,
            cv2.VideoWriter_fourcc(*'mp4v'), int(video_fps), (dashboard_w, 480)
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
        fps_display = video_fps

        print(f"[SYSTEM] Command Center online! Running at {video_fps:.1f} FPS native pacing. Press 'q' to quit.")

        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break

            start_time = time.time()
            frame = cv2.resize(frame, (640, 480))
            frame_count += 1

            zone_color = (0, 0, 255)
            detected_box = None
            conf_score = 0.0
            tracking_mode = "YOLO INFERENCE"

            # Day 18: Edge Optimization (YOLO every 5 frames; Tracker in-between)
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
                    tracker = create_tracker()
                    tracker.init(frame, detected_box)
                    tracking_active = True
                else:
                    tracking_active = False
            else:
                tracking_mode = "TRACKER"
                success, box = tracker.update(frame)
                if success:
                    detected_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                else:
                    tracking_active = False

            scx, scy = None, None
            if detected_box:
                x, y, w, h = detected_box
                raw_cx = int(x + w / 2)
                raw_cy = int(y + h / 2)

                # Smooth coordinates to remove frame-to-frame pixel jitter
                if self.smooth_x is None:
                    self.smooth_x, self.smooth_y = float(raw_cx), float(raw_cy)
                else:
                    self.smooth_x = 0.35 * raw_cx + 0.65 * self.smooth_x
                    self.smooth_y = 0.35 * raw_cy + 0.65 * self.smooth_y

                scx, scy = int(self.smooth_x), int(self.smooth_y)

                trail.append((scx, scy))
                if len(trail) > 30:
                    trail.pop(0)

                # Check collision and compute predictive velocity vector
                alert_triggered, frame = self.check_collision(scx, scy, trail, frame)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Eagle {conf_score:.2f}" if tracking_mode == "YOLO INFERENCE" else "Eagle (Tracked)"
                cv2.putText(frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (scx, scy), 5, (0, 255, 0), -1)

                self.write_telemetry(csv_writer, "Raptor", scx, scy)
            else:
                alert_triggered = (time.time() < self.brake_latch_until)

            if alert_triggered:
                print("Brake engaged: bird near blades")
                if int(time.time() * 6) % 2 == 0:
                    zone_color = (0, 255, 255)  # Flashing yellow warning on hazard zone

            # Draw turbine danger zone polygon with alpha blending
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.zone_pts], zone_color)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.polylines(frame, [self.zone_pts], True, zone_color, 2)

            # Draw historical motion trail
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 2)

            # Frame processing timing & FPS smoothing
            end_time = time.time()
            loop_time = end_time - start_time
            instant_fps = 1.0 / max(loop_time, 0.001)

            # For video playback, cap instant FPS to the video's rate
            if not is_webcam:
                instant_fps = min(instant_fps, video_fps)

            # Exponential Moving Average for smooth FPS readout
            fps_display = 0.90 * fps_display + 0.10 * instant_fps

            dashboard_frame = self.render_dashboard(
                frame, int(round(fps_display)), scx, scy, conf_score, tracking_mode, alert_triggered
            )

            out.write(dashboard_frame)
            cv2.imshow("Altamont Pass Bird Tracker", dashboard_frame)

            # Real-time frame pacing: wait exactly enough time to match video FPS
            if not is_webcam:
                elapsed_ms = (time.time() - start_time) * 1000.0
                delay_ms = max(1, int((frame_interval * 1000.0) - elapsed_ms))
            else:
                delay_ms = 1

            if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                break

        self.running = False
        producer_thread.join(timeout=2)
        cap.release()
        out.release()
        csv_file.close()
        cv2.destroyAllWindows()

        print("[SYSTEM] Clean shutdown complete. Generating flight trajectory analytics...")
        self.visualize_data()

if __name__ == "__main__":
    import sys
    default_source = "eagle_flying.mp4" if os.path.exists("eagle_flying.mp4") else 0
    video_arg = sys.argv[1] if len(sys.argv) > 1 else default_source

    if video_arg in ["0", "webcam", "camera"]:
        video_arg = 0

    system = RaptorDefenseSystem(video_source=video_arg)
    system.start()
