import math
import time
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

import config

class KalmanTrajectoryFilter:
    def __init__(self, dt: float = 1.0 / config.SENSOR_FPS):
        self.dt = dt
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition matrix F
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix H
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Covariance matrices
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(config.KALMAN_PROCESS_COV)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(config.KALMAN_MEASURE_COV)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False

    def reset(self):
        self.initialized = False

    def update(self, measured_x: float, measured_y: float) -> Tuple[float, float, float, float]:
        measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])
        if not self.initialized:
            self.kf.statePost = np.array([
                [np.float32(measured_x)],
                [np.float32(measured_y)],
                [0.0],
                [0.0]
            ], dtype=np.float32)
            self.initialized = True
            return measured_x, measured_y, 0.0, 0.0

        self.kf.predict()
        estimate = self.kf.correct(measurement)
        x = float(estimate[0][0])
        y = float(estimate[1][0])
        vx = float(estimate[2][0])
        vy = float(estimate[3][0])
        return x, y, vx, vy

class TortuosityGate:
    def __init__(self, window_size: int = config.TRAIL_MAX_POINTS):
        self.window_size = window_size

    def calculate(self, trail: List[Tuple[int, int]]) -> float:
        if len(trail) < 10:
            return 1.0

        pts = np.array(trail[-self.window_size:], dtype=np.float32)
        net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
        
        diffs = pts[1:] - pts[:-1]
        step_lengths = np.linalg.norm(diffs, axis=1)
        total_path = float(np.sum(step_lengths))

        if total_path < 5.0:
            return 1.0

        tortuosity = net_disp / total_path
        return max(0.0, min(1.0, tortuosity))

class TurbinePitchModel:
    def __init__(
        self,
        tau_actuator: float = config.TAU_ACTUATOR_S,
        pitch_rate_max: float = config.PITCH_RATE_MAX_DEG_S,
        pitch_operating: float = config.PITCH_OPERATING_DEG,
        pitch_feathered: float = config.PITCH_FEATHERED_DEG,
        nominal_rpm: float = config.NOMINAL_RPM
    ):
        self.tau = tau_actuator
        self.max_rate = pitch_rate_max
        self.pitch_operating = pitch_operating
        self.pitch_feathered = pitch_feathered
        self.nominal_rpm = nominal_rpm
        
        self.current_pitch = pitch_operating
        self.current_rpm = nominal_rpm
        self.last_update_time = time.time()

    def step(self, target_pitch: float, dt: Optional[float] = None) -> Tuple[float, float]:
        now = time.time()
        if dt is None:
            dt = max(1e-3, min(0.1, now - self.last_update_time))
        self.last_update_time = now

        # Actuator differential equation: tau * dbeta/dt + beta = target
        raw_rate = (target_pitch - self.current_pitch) / self.tau
        clamped_rate = max(-self.max_rate, min(self.max_rate, raw_rate))
        self.current_pitch += clamped_rate * dt
        self.current_pitch = max(self.pitch_operating, min(self.pitch_feathered, self.current_pitch))

        # Aerodynamic braking deceleration model
        fraction = (self.current_pitch - self.pitch_operating) / (self.pitch_feathered - self.pitch_operating)
        self.current_rpm = self.nominal_rpm * (1.0 - fraction)
        return self.current_pitch, self.current_rpm

class KinematicsEngine:
    def __init__(self):
        self.kalman = KalmanTrajectoryFilter()
        self.tortuosity_gate = TortuosityGate()
        self.turbine = TurbinePitchModel()
        self.brake_latch_until = 0.0

    def calculate_ttc(self, cx: float, cy: float, vx: float, vy: float) -> float:
        zone_center_x = (config.ZONE_MIN_X + config.ZONE_MAX_X) / 2.0
        zone_center_y = (config.ZONE_MIN_Y + config.ZONE_MAX_Y) / 2.0
        
        dx = zone_center_x - cx
        dy = zone_center_y - cy
        dist = math.hypot(dx, dy)
        speed = math.hypot(vx, vy)
        
        if speed < 1e-2:
            return 999.0

        # Dot product velocity projection toward zone
        v_toward = (vx * dx + vy * dy) / (dist + 1e-6)
        if v_toward <= 0:
            return 999.0
            
        ttc_frames = dist / v_toward
        ttc_seconds = ttc_frames / config.SENSOR_FPS
        return float(max(0.0, ttc_seconds))

    def evaluate_threat(
        self,
        raw_cx: int,
        raw_cy: int,
        trail: List[Tuple[int, int]],
        frame: np.ndarray,
        is_protected_eagle: bool = True
    ) -> Dict[str, Any]:
        # 1. 4-State Kalman Filter estimation
        smooth_x, smooth_y, vx, vy = self.kalman.update(raw_cx, raw_cy)
        scx, scy = int(round(smooth_x)), int(round(smooth_y))

        # 2. Flight Path Tortuosity Gate
        tortuosity = self.tortuosity_gate.calculate(trail)
        is_circling_thermal = (tortuosity < config.TORTUOSITY_THRESHOLD)

        # 3. Direct spatial breach check
        direct_breach = (
            config.ZONE_MIN_X <= scx <= config.ZONE_MAX_X and
            config.ZONE_MIN_Y <= scy <= config.ZONE_MAX_Y
        )

        # 4. Trajectory forward projection & line clipping
        fx = int(round(scx + vx * 20.0))
        fy = int(round(scy + vy * 20.0))
        predictive_hit, _, _ = cv2.clipLine(config.ZONE_RECT, (scx, scy), (fx, fy))

        # 5. Time-to-Collision (TTC)
        ttc = self.calculate_ttc(scx, scy, vx, vy)

        # 6. Curtailment Decision Logic Gate
        curtailment_required = False
        if is_protected_eagle:
            if direct_breach:
                curtailment_required = True
            elif predictive_hit and not is_circling_thermal:
                curtailment_required = True
            elif ttc <= config.TTC_WARNING_HORIZON_S and not is_circling_thermal:
                curtailment_required = True

        # Render predictive vector overlay
        if predictive_hit:
            color = (0, 140, 255) if not is_circling_thermal else (200, 200, 0)
            cv2.line(frame, (scx, scy), (fx, fy), color, 3)
            cv2.circle(frame, (fx, fy), 4, color, -1)
        else:
            cv2.line(frame, (scx, scy), (fx, fy), (0, 255, 0), 2)

        # Brake latching (hysteresis)
        now = time.time()
        if curtailment_required:
            self.brake_latch_until = now + config.BRAKE_HOLD_DURATION_S

        brake_active = (now < self.brake_latch_until)

        # Update turbine pitch model
        target_pitch = config.PITCH_FEATHERED_DEG if brake_active else config.PITCH_OPERATING_DEG
        pitch_deg, rpm = self.turbine.step(target_pitch)

        return {
            "smooth_x": scx,
            "smooth_y": scy,
            "vx": vx,
            "vy": vy,
            "tortuosity": tortuosity,
            "is_circling_thermal": is_circling_thermal,
            "direct_breach": direct_breach,
            "predictive_hit": predictive_hit,
            "ttc": ttc,
            "curtailment_required": curtailment_required,
            "brake_active": brake_active,
            "pitch_deg": pitch_deg,
            "rpm": rpm,
            "annotated_frame": frame
        }
