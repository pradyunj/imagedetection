import time
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class FramePacket:
    frame: Optional[np.ndarray]
    frame_idx: int
    arrival_time: float
    staleness_ms: float = 0.0

class BlockingFIFOBuffer:
    def __init__(self, maxsize: int = 10):
        self._queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(self, frame: Optional[np.ndarray], frame_idx: int = 0, timeout: float = 0.2) -> bool:
        packet = FramePacket(frame=frame, frame_idx=frame_idx, arrival_time=time.time())
        try:
            self._queue.put(packet, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout: float = 0.2) -> Optional[FramePacket]:
        try:
            packet = self._queue.get(timeout=timeout)
            if packet is not None:
                packet.staleness_ms = (time.time() - packet.arrival_time) * 1000.0
            return packet
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._queue.qsize()

    def full(self) -> bool:
        return self._queue.full()

class DropOldestRingBuffer:
    def __init__(self, maxsize: int = 10):
        self.maxsize = maxsize
        self._deque = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.dropped_count = 0

    def put(self, frame: Optional[np.ndarray], frame_idx: int = 0, timeout: float = 0.0) -> bool:
        packet = FramePacket(frame=frame, frame_idx=frame_idx, arrival_time=time.time())
        with self._cond:
            if len(self._deque) == self.maxsize:
                self._deque.popleft()
                self.dropped_count += 1
            self._deque.append(packet)
            self._cond.notify()
        return True

    def get(self, timeout: float = 0.2) -> Optional[FramePacket]:
        with self._cond:
            if not self._deque:
                if not self._cond.wait(timeout=timeout):
                    return None
            if self._deque:
                packet = self._deque.popleft()
                packet.staleness_ms = (time.time() - packet.arrival_time) * 1000.0
                return packet
            return None

    def qsize(self) -> int:
        with self._lock:
            return len(self._deque)

    def full(self) -> bool:
        with self._lock:
            return len(self._deque) == self.maxsize

class FramePacer:
    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps if target_fps > 0 else 30.0
        self.interval_s = 1.0 / self.target_fps

    def wait_remaining(self, start_time: float) -> float:
        elapsed = time.time() - start_time
        remaining = self.interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        return max(0.0, remaining)
