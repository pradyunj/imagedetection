"""
Altamont Pass Project - Autonomous Raptor Detection & Turbine Curtailment
Combines configuration, frame buffers, YOLOv8 + Hungarian/OpenCV tracking,
4-state Kalman filter kinematics, tortuosity gate, SCADA pitch ODE, and HUD.
"""

import argparse
import collections
import csv
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np

# ============================================================================
# 1. SYSTEM CONFIGURATION
# ============================================================================

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_VIDEO_PATH = "eagle_flying.mp4"

MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.30

# Taxonomy Mapping
# Class 0: Protected Golden Eagle / Bald Eagle (Triggers Curtailment)
# Class 1: Turkey Vulture (Non-protected sympathetic raptor - Suppressed)
# Class 2: Common Raven (Non-protected sympathetic species - Suppressed)
# Class 3: Other Bird / Background (Suppressed)
TAXONOMY_MAP = {
    0: {"name": "Golden Eagle", "protected": True, "class_id": 0},
    1: {"name": "Turkey Vulture", "protected": False, "class_id": 1},
    2: {"name": "Common Raven", "protected": False, "class_id": 2},
    3: {"name": "Other Bird", "protected": False, "class_id": 3},
}
PROTECTED_CLASS_ID = 0

# Turbine Hazard Zone Geometry (in 1280x720 space)
TURBINE_CENTER = (640, 360)
ROTOR_RADIUS_PX = 250

# Time-to-Collision (TTC) & Tortuosity Gate Thresholds
TTC_CRITICAL_THRESHOLD_S = 15.0
TTC_WARNING_HORIZON_S = 30.0
TORTUOSITY_WINDOW_FRAMES = 30
TORTUOSITY_THRESHOLD = 0.45

# Wind Turbine Physical Pitch ODE Parameters
PITCH_OPERATING_DEG = 0.0
PITCH_FEATHERED_DEG = 90.0
PITCH_RATE_MAX_DEG_PER_S = 8.0
PITCH_TIME_CONSTANT_TAU = 1.5

NOMINAL_RPM = 18.0
FEATHERED_RPM = 0.5

# Hardware Serial Communications
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUDRATE = 115200
MOCK_HARDWARE = True


# ============================================================================
# 2. FRAME CONCURRENCY BUFFERS & PACING
# ============================================================================

class FrameBuffer:
    def __init__(self, capacity: int = 30, buffer_type: str = "ring"):
        self.capacity = capacity
        self.buffer_type = buffer_type.lower()
        if self.buffer_type == "ring":
            self.buffer = collections.deque(maxlen=capacity)
        else:
            self.buffer = collections.deque()

    def put(self, frame: np.ndarray) -> None:
        if self.buffer_type == "fifo" and len(self.buffer) >= self.capacity:
            self.buffer.popleft()
        self.buffer.append(frame)

    def get(self) -> Optional[np.ndarray]:
        if len(self.buffer) == 0:
            return None
        return self.buffer.popleft()

    def __len__(self) -> int:
        return len(self.buffer)


class FramePacer:
    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.frame_time = 1.0 / max(1.0, target_fps)
        self.last_time = time.time()

    def sleep(self) -> None:
        elapsed = time.time() - self.last_time
        sleep_dur = self.frame_time - elapsed
        if sleep_dur > 0:
            time.sleep(sleep_dur)
        self.last_time = time.time()


# ============================================================================

from core.detector import RaptorDetector
from core.kinematics import TrajectoryKinematics
from hardware.scada_serial import SCADAController
from ui.dashboard import DashboardRenderer
 ============================================================================
# 7. MAIN RUNTIME LOOP & CSV TELEMETRY LOGGER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Altamont Pass Raptor Detection & Curtailment System")
    parser.add_argument("source", nargs="?", default=DEFAULT_VIDEO_PATH, help="Video source (path, 'webcam', or index)")
    parser.add_argument("--buffer", choices=["fifo", "ring"], default="ring", help="Concurrency frame buffer type")
    parser.add_argument("--modulo", type=int, default=5, help="Run YOLO every Nth frame")
    parser.add_argument("--tracker", choices=["MIL", "CSRT", "KCF", "MOSSE"], default="MIL", help="Tracker algorithm")
    parser.add_argument("--log", default="experiments/results/telemetry_latest.csv", help="Output CSV log path")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run until 'q')")
    parser.add_argument("--headless", action="store_true", help="Disable GUI window for headless runs")
    return parser.parse_args()


def main():
    args = parse_args()
    source_val = 0 if args.source == "webcam" else (int(args.source) if args.source.isdigit() else args.source)

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = DEFAULT_FPS

    frame_buffer = FrameBuffer(capacity=30, buffer_type=args.buffer)
    frame_pacer = FramePacer(target_fps=fps)
    detector = RaptorDetector(modulo=args.modulo, tracker_type=args.tracker)
    kinematics = TrajectoryKinematics(fps=fps)
    scada = SCADAController(port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE, mock=MOCK_HARDWARE)
    dashboard = DashboardRenderer()
    scada.connect()

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    csv_file = open(args.log, mode="w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "frame", "timestamp", "class_name", "confidence", "cx", "cy",
        "vx", "vy", "tortuosity", "ttc", "pitch_deg", "rpm", "curtailing"
    ])
    csv_writer.writeheader()

    frame_index = 0
    prevented_count = 0
    was_curtailing = False

    print(f"[INFO] Running Altamont Pass Monitor | Source: {args.source} | Modulo: {args.modulo} | Tracker: {args.tracker}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source_val, str) and args.source != "webcam" and args.max_frames == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_buffer.put(frame)
            current_frame = frame_buffer.get()
            if current_frame is None:
                continue

            frame_index += 1
            dt = 1.0 / fps

            target_info = detector.process_frame(current_frame, frame_index)
            threat_info = None

            if target_info and target_info.get("detected"):
                is_protected = target_info.get("is_eagle", True)
                cx, cy = target_info["cx"], target_info["cy"]
                threat_info = kinematics.update(cx, cy, is_protected=is_protected, dt=dt)

                if is_protected and threat_info["should_curtail"]:
                    scada.command_pitch(PITCH_FEATHERED_DEG)
                    if not was_curtailing:
                        prevented_count += 1
                        was_curtailing = True
                else:
                    scada.command_pitch(PITCH_OPERATING_DEG)
                    was_curtailing = False
            else:
                kinematics.reset()
                scada.command_pitch(PITCH_OPERATING_DEG)
                was_curtailing = False

            scada.update_physics(dt)
            pitch_deg = scada.get_current_pitch()
            rpm = scada.get_current_rpm()

            if threat_info:
                threat_info["pitch_deg"] = pitch_deg
                threat_info["rpm"] = rpm

            csv_writer.writerow({
                "frame": frame_index,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "class_name": target_info.get("class_name", "None") if target_info else "None",
                "confidence": round(target_info.get("confidence", 0.0), 4) if target_info else 0.0,
                "cx": target_info.get("cx", -1) if target_info else -1,
                "cy": target_info.get("cy", -1) if target_info else -1,
                "vx": round(threat_info.get("vx", 0.0), 2) if threat_info else 0.0,
                "vy": round(threat_info.get("vy", 0.0), 2) if threat_info else 0.0,
                "tortuosity": round(threat_info.get("tortuosity", 1.0), 4) if threat_info else 1.0,
                "ttc": round(threat_info.get("ttc", 999.0), 2) if threat_info else 999.0,
                "pitch_deg": round(pitch_deg, 2),
                "rpm": round(rpm, 2),
                "curtailing": int(was_curtailing),
            })

            if not args.headless:
                output_frame = dashboard.render(
                    current_frame,
                    fps=int(fps),
                    tracking_mode=f"YOLO + {args.tracker}",
                    target_info=target_info,
                    threat_info=threat_info,
                    prevented_count=prevented_count,
                )
                cv2.imshow("Altamont Pass Raptor Monitor", output_frame)
                frame_pacer.sleep()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

    finally:
        csv_file.close()
        cap.release()
        scada.close()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[INFO] Shutdown complete. Telemetry saved to {args.log}")


if __name__ == "__main__":
    main()

