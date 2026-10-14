import os
import sys
import time
import csv
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from core.buffer import BlockingFIFOBuffer, DropOldestRingBuffer, FramePacer, FramePacket
from core.detector import MultiTargetDetector, TrackedTarget
from core.kinematics import KinematicsEngine
from hardware.scada_serial import ScadaSerialController
from ui.dashboard import DashboardRenderer

class RaptorDefenseSystem:
    def __init__(
        self,
        video_source: Any = config.DEFAULT_VIDEO_SOURCE,
        buffer_type: str = "fifo",
        modulo_n: int = 5,
        tracker_type: str = "MIL",
        mock_scada: bool = True
    ):
        if isinstance(video_source, str) and not os.path.exists(video_source):
            print(f"[SYSTEM] Notice: Video asset '{video_source}' not found. Defaulting to webcam 0.")
            self.video_source = 0
        else:
            self.video_source = video_source

        self.is_webcam = isinstance(self.video_source, int)
        self.running = True

        # Concurrency frame buffer
        if buffer_type.lower() == "ring":
            self.buffer = DropOldestRingBuffer(maxsize=10)
        else:
            self.buffer = BlockingFIFOBuffer(maxsize=10)

        # Core scientific engines
        model_weights = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
        self.detector = MultiTargetDetector(
            model_path=model_weights,
            modulo_n=modulo_n,
            tracker_type=tracker_type,
            conf_threshold=0.35
        )
        self.kinematics = KinematicsEngine()
        self.scada = ScadaSerialController(mock=mock_scada)
        self.dashboard = DashboardRenderer(
            sidebar_width=config.SIDEBAR_WIDTH,
            height=config.FRAME_HEIGHT
        )

        self.trail: List[tuple] = []
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.RECORDINGS_DIR, exist_ok=True)

    def producer_worker(self, cap: cv2.VideoCapture):
        frame_idx = 0
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break
            packet = FramePacket(frame=frame, timestamp=time.time(), frame_index=frame_idx)
            self.buffer.put(packet, block=(not self.is_webcam), timeout=0.2)
            frame_idx += 1
        self.buffer.put(None, block=False)

    def draw_hazard_zone(self, frame: np.ndarray, alert_active: bool) -> np.ndarray:
        zone_color = (0, 0, 255)
        if alert_active and int(time.time() * 6) % 2 == 0:
            zone_color = (0, 255, 255)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [config.ZONE_POLYGON], zone_color)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.polylines(frame, [config.ZONE_POLYGON], True, zone_color, 2)
        return frame

    def draw_overlays(
        self,
        frame: np.ndarray,
        targets: List[TrackedTarget],
        alert_active: bool
    ) -> np.ndarray:
        frame = self.draw_hazard_zone(frame, alert_active)

        for target in targets:
            x, y, w, h = target.bbox
            if target.class_id == config.CLASS_EAGLE_PROTECTED:
                box_color = (0, 0, 255) if alert_active else (0, 255, 0)
            elif target.class_id == config.CLASS_SYMPATRIC_VULTURE:
                box_color = (0, 165, 255)
            elif target.class_id == config.CLASS_SYMPATRIC_CORVID:
                box_color = (255, 255, 0)
            else:
                box_color = (200, 200, 200)

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            label = f"{target.class_name} {target.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                box_color,
                2
            )

        for i in range(1, len(self.trail)):
            cv2.line(frame, self.trail[i - 1], self.trail[i], (0, 255, 0), 2)

        return frame

    def visualize_trajectory(self, csv_path: str):
        if not os.path.exists(csv_path):
            return
        try:
            df = pd.read_csv(csv_path)
            if df.empty or 'Smooth_X' not in df.columns:
                return

            df['X_Filtered'] = df['Smooth_X'].rolling(window=7, min_periods=1, center=True).mean()
            df['Y_Filtered'] = df['Smooth_Y'].rolling(window=7, min_periods=1, center=True).mean()
            sample_df = df.iloc[::4]

            plt.figure(figsize=(10, 6))
            plt.plot(df['X_Filtered'], df['Y_Filtered'], color='crimson', linewidth=2.5, label='Filtered Raptor Trajectory')
            plt.scatter(sample_df['X_Filtered'], sample_df['Y_Filtered'], color='darkred', s=20, zorder=3)
            plt.gca().add_patch(
                plt.Rectangle(
                    (config.ZONE_MIN_X, config.ZONE_MIN_Y),
                    config.ZONE_MAX_X - config.ZONE_MIN_X,
                    config.ZONE_MAX_Y - config.ZONE_MIN_Y,
                    color='royalblue',
                    alpha=0.25,
                    label='Turbine Hazard Airspace'
                )
            )

            plt.title('Altamont Pass Autonomous Raptor Defense: Trajectory Telemetry', fontsize=12, fontweight='bold')
            plt.xlabel('Camera X Coordinate (Pixels)', fontsize=10)
            plt.ylabel('Camera Y Coordinate (Pixels)', fontsize=10)
            plt.xlim(0, config.FRAME_WIDTH)
            plt.ylim(0, config.FRAME_HEIGHT)
            plt.gca().invert_yaxis()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='upper right')
            plt.tight_layout()

            plot_path = os.path.join(config.RECORDINGS_DIR, 'latest_trajectory_plot.png')
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"[SYSTEM] Trajectory plot saved: {plot_path}")
        except Exception as e:
            print(f"[SYSTEM] Trajectory plotting notice: {e}")

    def start(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {self.video_source}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0

        pacer = FramePacer(fps=video_fps)
        total_width = config.FRAME_WIDTH + config.SIDEBAR_WIDTH

        video_out_path = os.path.join(config.RECORDINGS_DIR, f"{timestamp}_video_processing.mp4")
        writer = cv2.VideoWriter(
            video_out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            int(video_fps),
            (total_width, config.FRAME_HEIGHT)
        )

        csv_path = os.path.join(config.LOG_DIR, f"{timestamp}_telemetry.csv")
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "Timestamp", "Frame_Idx", "Class_Name", "Conf",
            "Raw_X", "Raw_Y", "Smooth_X", "Smooth_Y",
            "VX", "VY", "Tortuosity", "TTC", "Pitch_Deg", "Rotor_RPM", "Brake_Active"
        ])

        producer_thread = threading.Thread(target=self.producer_worker, args=(cap,))
        producer_thread.daemon = True
        producer_thread.start()

        fps_display = video_fps
        frame_idx = 0
        print(f"[SYSTEM] Command Center online at {video_fps:.1f} FPS native pacing. Press 'q' to quit.")

        try:
            while True:
                packet = self.buffer.get(block=True)
                if packet is None:
                    break

                t_start = time.time()
                frame = cv2.resize(packet.frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                frame_idx += 1

                # 1. Multi-target detection and tracking pass
                targets, tracking_mode = self.detector.update(frame, frame_idx)
                primary = self.detector.get_primary_target()

                # 2. Kinematics and physics evaluation
                if primary is not None:
                    raw_cx, raw_cy = primary.center
                    self.trail.append((raw_cx, raw_cy))
                    if len(self.trail) > config.TRAIL_MAX_POINTS:
                        self.trail.pop(0)

                    is_eagle = (primary.class_id == config.CLASS_EAGLE_PROTECTED)
                    threat = self.kinematics.evaluate_threat(
                        raw_cx=raw_cx,
                        raw_cy=raw_cy,
                        trail=self.trail,
                        frame=frame,
                        is_protected_eagle=is_eagle
                    )
                    target_info = {
                        "detected": True,
                        "class_name": primary.class_name,
                        "confidence": primary.confidence,
                        "cx": threat["smooth_x"],
                        "cy": threat["smooth_y"],
                        "is_eagle": is_eagle
                    }
                else:
                    threat = self.kinematics.evaluate_idle(frame)
                    target_info = None

                # 3. Synchronize with hardware SCADA
                if threat["brake_active"]:
                    self.scada.emergency_feather()
                else:
                    self.scada.send_blade_pitch(threat["pitch_deg"])

                # 4. Render overlays and HUD
                frame = self.draw_overlays(frame, targets, threat["brake_active"])

                t_loop = max(1e-4, time.time() - t_start)
                instant_fps = min(video_fps, 1.0 / t_loop) if not self.is_webcam else (1.0 / t_loop)
                fps_display = 0.90 * fps_display + 0.10 * instant_fps

                dashboard = self.dashboard.render(
                    frame=frame,
                    fps=int(round(fps_display)),
                    tracking_mode=tracking_mode,
                    target_info=target_info,
                    threat_info=threat,
                    prevented_count=self.detector.prevented_curtailments_count
                )

                writer.write(dashboard)
                cv2.imshow("Altamont Pass Autonomous Raptor Defense", dashboard)

                # Log telemetry
                if primary is not None:
                    csv_writer.writerow([
                        datetime.now().isoformat(), frame_idx, primary.class_name, f"{primary.confidence:.2f}",
                        raw_cx, raw_cy, threat["smooth_x"], threat["smooth_y"],
                        f"{threat['vx']:.2f}", f"{threat['vy']:.2f}",
                        f"{threat['tortuosity']:.3f}", f"{threat['ttc']:.2f}",
                        f"{threat['pitch_deg']:.2f}", f"{threat['rpm']:.2f}",
                        int(threat["brake_active"])
                    ])

                delay_ms = pacer.get_wait_key_delay(t_loop, is_live=self.is_webcam)
                if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                    break
        finally:
            self.running = False
            producer_thread.join(timeout=1.5)
            cap.release()
            writer.release()
            csv_file.close()
            self.scada.close()
            cv2.destroyAllWindows()

        print("[SYSTEM] Mission complete. Generating post-flight analytics...")
        self.visualize_trajectory(csv_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Altamont Pass Autonomous Raptor Defense System")
    parser.add_argument("source", nargs="?", default=config.DEFAULT_VIDEO_SOURCE, help="Video file or 'webcam' / 0")
    parser.add_argument("--buffer", choices=["fifo", "ring"], default="fifo", help="Frame buffer architecture")
    parser.add_argument("--modulo", type=int, default=5, help="Detection modulo interval N")
    parser.add_argument("--tracker", choices=["MIL", "KCF", "CSRT", "MOSSE"], default="MIL", help="Tracker type")
    parser.add_argument("--no-scada", action="store_true", help="Disable SCADA controller")
    args = parser.parse_args()

    src = 0 if args.source in ["0", "webcam", "camera"] else args.source
    system = RaptorDefenseSystem(
        video_source=src,
        buffer_type=args.buffer,
        modulo_n=args.modulo,
        tracker_type=args.tracker,
        mock_scada=not args.no_scada
    )
    system.start()
