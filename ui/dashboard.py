import time
from typing import Optional, Dict, Any
import numpy as np
import cv2

import config

class DashboardRenderer:
    def __init__(self, sidebar_width: int = 280, height: int = config.FRAME_HEIGHT):
        self.sidebar_width = sidebar_width
        self.height = height

    def render(
        self,
        frame: np.ndarray,
        fps: int,
        tracking_mode: str,
        target_info: Optional[Dict[str, Any]],
        threat_info: Optional[Dict[str, Any]],
        prevented_count: int = 0
    ) -> np.ndarray:
        sidebar = np.zeros((self.height, self.sidebar_width, 3), dtype=np.uint8)

        # 1. Header
        cv2.putText(sidebar, "ALTAMONT PASS", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(sidebar, "TURBINE RESEARCH MONITOR", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        cv2.line(sidebar, (18, 58), (self.sidebar_width - 18, 58), (70, 70, 70), 1)

        # 2. Basic Telemetry
        cv2.putText(sidebar, f"FPS: {fps}", (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(sidebar, f"MODE: {tracking_mode}", (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        status_text = "SCANNING..."
        status_color = (120, 120, 120)
        coord_text = "X: ---  Y: ---"
        conf_text = "CONF: N/A"

        if target_info and target_info.get("detected"):
            name = target_info.get("class_name", "Target")
            conf = target_info.get("confidence", 0.0)
            cx = target_info.get("cx")
            cy = target_info.get("cy")
            status_text = f"{name.upper()}"
            status_color = (0, 255, 0) if target_info.get("is_eagle", True) else (0, 200, 255)
            coord_text = f"X: {cx}  Y: {cy}"
            conf_text = f"CONF: {conf:.2f}"

        cv2.putText(sidebar, f"TARGET: {status_text}", (18, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1)
        cv2.putText(sidebar, coord_text, (18, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
        cv2.putText(sidebar, conf_text, (18, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
        cv2.line(sidebar, (18, 172), (self.sidebar_width - 18, 172), (70, 70, 70), 1)

        # 3. ISEF Kinematics & Physics Metrics
        tortuosity = threat_info.get("tortuosity", 1.0) if threat_info else 1.0
        ttc = threat_info.get("ttc", 999.0) if threat_info else 999.0
        pitch_deg = threat_info.get("pitch_deg", config.PITCH_OPERATING_DEG) if threat_info else config.PITCH_OPERATING_DEG
        rpm = threat_info.get("rpm", config.NOMINAL_RPM) if threat_info else config.NOMINAL_RPM

        # TTC Display
        ttc_str = f"{ttc:.1f}s" if ttc < 99.0 else "> 30s"
        ttc_color = (0, 0, 255) if ttc <= config.TTC_WARNING_HORIZON_S else (255, 255, 255)
        cv2.putText(sidebar, f"TTC: {ttc_str}", (18, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ttc_color, 1)

        # Tortuosity Display
        tort_color = (0, 220, 255) if tortuosity < config.TORTUOSITY_THRESHOLD else (255, 255, 255)
        tort_label = f"TORTUOSITY: {tortuosity:.2f}"
        if tortuosity < config.TORTUOSITY_THRESHOLD:
            tort_label += " (SOARING)"
        cv2.putText(sidebar, tort_label, (18, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tort_color, 1)

        # Rotor RPM
        cv2.putText(sidebar, f"ROTOR RPM: {rpm:.1f}", (18, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        # Blade Pitch Gauge
        cv2.putText(sidebar, f"BLADE PITCH: {pitch_deg:.1f} deg", (18, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        gauge_x = 18
        gauge_y = 265
        gauge_w = self.sidebar_width - 36
        gauge_h = 10
        cv2.rectangle(sidebar, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (50, 50, 50), -1)

        fraction = (pitch_deg - config.PITCH_OPERATING_DEG) / (config.PITCH_FEATHERED_DEG - config.PITCH_OPERATING_DEG)
        fill_w = int(max(0.0, min(1.0, fraction)) * gauge_w)
        gauge_color = (0, int(255 * (1.0 - fraction)), int(255 * fraction))
        if fill_w > 0:
            cv2.rectangle(sidebar, (gauge_x, gauge_y), (gauge_x + fill_w, gauge_y + gauge_h), gauge_color, -1)
        cv2.rectangle(sidebar, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (120, 120, 120), 1)

        # Prevented False Alarms Counter
        cv2.putText(sidebar, f"PREVENTED CURTAIL: {prevented_count}", (18, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1)
        cv2.line(sidebar, (18, 308), (self.sidebar_width - 18, 308), (70, 70, 70), 1)

        # 4. Large Industrial Status Button
        btn_center = (self.sidebar_width // 2, 385)
        btn_radius = 46
        alert_active = threat_info.get("brake_active", False) if threat_info else False

        if alert_active:
            pulse = int(180 + 75 * np.sin(time.time() * 8))
            cv2.circle(sidebar, btn_center, btn_radius, (0, 0, pulse), -1)
            cv2.circle(sidebar, btn_center, btn_radius + 4, (0, 0, 255), 2)
            cv2.putText(sidebar, "BRAKE", (btn_center[0] - 27, btn_center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(sidebar, "ENGAGED", (btn_center[0] - 38, btn_center[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        else:
            cv2.circle(sidebar, btn_center, btn_radius, (0, 160, 0), -1)
            cv2.circle(sidebar, btn_center, btn_radius + 4, (0, 255, 0), 2)
            cv2.putText(sidebar, "ALL", (btn_center[0] - 17, btn_center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(sidebar, "CLEAR", (btn_center[0] - 28, btn_center[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        return np.hstack([frame, sidebar])
