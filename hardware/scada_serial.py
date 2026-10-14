import time
import threading
from typing import Optional, Tuple
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

import config

class ScadaSerialController:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[Any] = None
        self.simulated = True
        
        self.tachometer_rpm = config.NOMINAL_RPM
        self.telemetry_pitch = config.PITCH_OPERATING_DEG
        self.status_msg = "SIMULATED"
        self._running = False
        self._lock = threading.Lock()

        self._connect()

    def _find_arduino_port(self) -> Optional[str]:
        if not SERIAL_AVAILABLE:
            return None
        ports = serial.tools.list_ports.comports()
        for p in ports:
            p_desc = (p.description or "").lower()
            if "arduino" in p_desc or "ch340" in p_desc or "usb serial" in p_desc or "usbmodem" in p_desc:
                return p.device
        return None

    def _connect(self):
        if not SERIAL_AVAILABLE:
            self.simulated = True
            return

        target_port = self.port or self._find_arduino_port()
        if target_port:
            try:
                self.ser = serial.Serial(target_port, self.baudrate, timeout=self.timeout)
                self.simulated = False
                self.status_msg = f"CONNECTED ({target_port})"
                self._running = True
                threading.Thread(target=self._read_worker, daemon=True).start()
                return
            except Exception:
                pass
        
        self.simulated = True
        self.status_msg = "SIMULATED RIG"

    def _read_worker(self):
        while self._running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("$TELEM"):
                    parts = line.split(',')
                    for p in parts[1:]:
                        if '=' in p:
                            k, v = p.split('=', 1)
                            with self._lock:
                                if k == "RPM":
                                    self.tachometer_rpm = float(v)
                                elif k == "PITCH":
                                    self.telemetry_pitch = float(v)
            except Exception:
                break

    def send_pitch_command(self, pitch_deg: float, brake_active: bool) -> bool:
        cmd_str = f"$CMD,PITCH={pitch_deg:.1f},BRAKE={1 if brake_active else 0}\n"
        if not self.simulated and self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd_str.encode('utf-8'))
                return True
            except Exception:
                self.simulated = True
                self.status_msg = "FALLBACK SIMULATED"
        
        # Software simulation feedback
        with self._lock:
            self.telemetry_pitch = pitch_deg
            fraction = (pitch_deg - config.PITCH_OPERATING_DEG) / (config.PITCH_FEATHERED_DEG - config.PITCH_OPERATING_DEG)
            self.tachometer_rpm = config.NOMINAL_RPM * max(0.0, 1.0 - fraction)
        return True

    def get_telemetry(self) -> Tuple[float, float, str]:
        with self._lock:
            return self.tachometer_rpm, self.telemetry_pitch, self.status_msg

    def close(self):
        self._running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
