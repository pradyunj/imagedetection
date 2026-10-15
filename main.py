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
# 3. DETECTOR, IOU HUNGARIAN MATCHING & MODULO TRACKING
# ============================================================================

def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    return float(inter_area / (area_a + area_b - inter_area + 1e-6))


class RaptorDetector:
    def __init__(self, modulo: int = 5, tracker_type: str = "MIL"):
        self.modulo = max(1, modulo)
        self.tracker_type = tracker_type
        self.tracker = None
        self.tracking_active = False
        self.last_bbox = None
        self.class_id = PROTECTED_CLASS_ID
        self.confidence = 0.0
        self.is_coco_model = False

        try:
            from ultralytics import YOLO
            self.model = YOLO(MODEL_PATH)
            self.has_yolo = True
            names = getattr(self.model, "names", {})
            if isinstance(names, dict) and names.get(14) == "bird":
                self.is_coco_model = True
        except Exception:
            self.has_yolo = False

    def _create_tracker(self):
        if self.tracker_type == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if self.tracker_type == "KCF" and hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if self.tracker_type == "MOSSE" and hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        return cv2.TrackerMIL_create()

    def _select_best_detection(self, candidates: list) -> Optional[dict]:
        if not candidates:
            return None
        if self.last_bbox is not None and len(candidates) > 1:
            cost_matrix = np.array([[1.0 - compute_iou(self.last_bbox, c["bbox"])] for c in candidates])
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, _ = linear_sum_assignment(cost_matrix)
                best_idx = int(row_ind[0])
            except Exception:
                best_idx = int(np.argmin(cost_matrix[:, 0]))
            if 1.0 - cost_matrix[best_idx, 0] >= IOU_THRESHOLD:
                return candidates[best_idx]
        return max(candidates, key=lambda c: c["conf"])

    def process_frame(self, frame: np.ndarray, frame_index: int) -> dict:
        h, w = frame.shape[:2]
        run_detector = (frame_index % self.modulo == 1) or not self.tracking_active

        if run_detector:
            candidates = []
            if self.has_yolo:
                results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                for r in results:
                    for box in r.boxes:
                        raw_cid = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if self.is_coco_model:
                            if raw_cid != 14:
                                continue
                            cid = PROTECTED_CLASS_ID
                        else:
                            cid = raw_cid if raw_cid in TAXONOMY_MAP else 3
                        xywh = box.xywh[0].cpu().numpy()
                        bbox = (
                            int(xywh[0] - xywh[2] / 2),
                            int(xywh[1] - xywh[3] / 2),
                            int(xywh[2]),
                            int(xywh[3]),
                        )
                        candidates.append({"bbox": bbox, "cid": cid, "conf": conf})
            else:
                candidates.append({
                    "bbox": (int(w * 0.4), int(h * 0.3), 80, 60),
                    "cid": PROTECTED_CLASS_ID,
                    "conf": 0.88,
                })

            chosen = self._select_best_detection(candidates)
            if chosen is not None:
                self.tracker = self._create_tracker()
                self.tracker.init(frame, chosen["bbox"])
                self.tracking_active = True
                self.last_bbox = chosen["bbox"]
                self.class_id = chosen["cid"]
                self.confidence = chosen["conf"]
            else:
                self.tracking_active = False

        elif self.tracking_active and self.tracker is not None:
            success, bbox = self.tracker.update(frame)
            if success:
                self.last_bbox = tuple(int(v) for v in bbox)
            else:
                self.tracking_active = False

        if self.tracking_active and self.last_bbox is not None:
            x, y, bw, bh = self.last_bbox
            cx = x + bw // 2
            cy = y + bh // 2
            tax_info = TAXONOMY_MAP.get(self.class_id, TAXONOMY_MAP[0])
            return {
                "detected": True,
                "bbox": (x, y, bw, bh),
                "cx": cx,
                "cy": cy,
                "class_id": self.class_id,
                "class_name": tax_info["name"],
                "is_eagle": tax_info["protected"],
                "confidence": self.confidence,
            }

        return {"detected": False}


# ============================================================================
# 4. 4-STATE KALMAN FILTER, TORTUOSITY GATE & 2D RAY-CIRCLE TTC
# ============================================================================

class TrajectoryKinematics:
    def __init__(self, fps: float = 30.0):
        self.fps = max(1.0, fps)
        self.history = collections.deque(maxlen=TORTUOSITY_WINDOW_FRAMES)
        self.kf = self._init_kalman(1.0 / self.fps)
        self.initialized = False

    def _init_kalman(self, dt: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def reset(self) -> None:
        self.history.clear()
        self.initialized = False

    def compute_tortuosity(self) -> float:
        if len(self.history) < 5:
            return 1.0
        pts = np.array(self.history, dtype=np.float32)
        step_distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_path_length = float(np.sum(step_distances))
        if total_path_length < 1e-5:
            return 1.0
        straight_line_dist = float(np.linalg.norm(pts[-1] - pts[0]))
        return float(np.clip(straight_line_dist / total_path_length, 0.0, 1.0))

    def _compute_vector_ttc(self, pos: np.ndarray, vel: np.ndarray) -> float:
        center = np.array(TURBINE_CENTER, dtype=np.float32)
        rel = pos - center
        dist = float(np.linalg.norm(rel))
        if dist <= ROTOR_RADIUS_PX:
            return 0.0

        speed_sq = float(np.dot(vel, vel))
        if speed_sq < 1e-3:
            return 999.0

        # Check if velocity has a closing component toward the rotor center
        proj = float(np.dot(rel, vel))
        if proj >= 0.0:
            return 999.0

        # Solve quadratic |rel + vel * t|^2 = R^2 for exact ray-circle entry
        a = speed_sq
        b = 2.0 * proj
        c = dist * dist - float(ROTOR_RADIUS_PX ** 2)
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            t_hit = (-b - np.sqrt(disc)) / (2.0 * a)
            if t_hit >= 0.0:
                return float(t_hit)

        # Fallback to radial closing speed when heading generally toward the rotor zone
        closing_speed = -proj / dist
        return float(max(0.0, (dist - ROTOR_RADIUS_PX) / max(closing_speed, 1e-3)))

    def update(self, cx: int, cy: int, is_protected: bool, dt: float = 0.033) -> dict:
        dt = max(1e-3, dt)
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kf.statePost = np.array([[np.float32(cx)], [np.float32(cy)], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True

        self.kf.predict()
        state = self.kf.correct(meas)
        filt_x, filt_y, vx, vy = [float(v) for v in state.flatten()]

        self.history.append((filt_x, filt_y))
        tortuosity = self.compute_tortuosity()

        pos_vec = np.array([filt_x, filt_y], dtype=np.float32)
        vel_vec = np.array([vx, vy], dtype=np.float32)
        dist_to_center = float(np.linalg.norm(pos_vec - np.array(TURBINE_CENTER, dtype=np.float32)))
        ttc = self._compute_vector_ttc(pos_vec, vel_vec)

        is_direct = tortuosity >= TORTUOSITY_THRESHOLD
        should_curtail = bool(
            is_protected and is_direct and (ttc <= TTC_CRITICAL_THRESHOLD_S or dist_to_center <= ROTOR_RADIUS_PX)
        )

        return {
            "x": filt_x,
            "y": filt_y,
            "vx": vx,
            "vy": vy,
            "tortuosity": tortuosity,
            "is_direct": is_direct,
            "ttc": ttc,
            "dist_px": dist_to_center,
            "should_curtail": should_curtail,
        }


# ============================================================================

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

