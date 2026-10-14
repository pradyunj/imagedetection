import time
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np

@dataclass
class FramePacket:
    frame: Optional[np.ndarray]
    frame_idx: int = 0
    arrival_time: float = 0.0
    timestamp: float = 0.0
    frame_index: int = 0
    staleness_ms: float = 0.0

    def __post_init__(self):
        if self.arrival_time == 0.0 and self.timestamp != 0.0:
            self.arrival_time = self.timestamp
        elif self.timestamp == 0.0 and self.arrival_time != 0.0:
            self.timestamp = self.arrival_time
        if self.frame_idx == 0 and self.frame_index != 0:
            self.frame_idx = self.frame_index
        elif self.frame_index == 0 and self.frame_idx != 0:
            self.frame_index = self.frame_idx

class BlockingFIFOBuffer:
    def __init__(self, maxsize: int = 10):
        self._queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(
        self,
        item: Any,
        frame_idx: int = 0,
        timeout: float = 0.2,
        block: bool = True
    ) -> bool:
        if isinstance(item, FramePacket):
            packet = item
        elif item is None:
            packet = None
        else:
            packet = FramePacket(frame=item, frame_idx=frame_idx, arrival_time=time.time())

        try:
            self._queue.put(packet, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout: float = 0.2, block: bool = True) -> Optional[FramePacket]:
        try:
            packet = self._queue.get(block=block, timeout=timeout)
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

    def put(
        self,
        item: Any,
        frame_idx: int = 0,
        timeout: float = 0.0,
        block: bool = False
    ) -> bool:
        if isinstance(item, FramePacket):
            packet = item
        elif item is None:
            packet = None
        else:
            packet = FramePacket(frame=item, frame_idx=frame_idx, arrival_time=time.time())

        with self._cond:
            if len(self._deque) == self.maxsize:
                self._deque.popleft()
                self.dropped_count += 1
            self._deque.append(packet)
            self._cond.notify()
        return True

    def get(self, timeout: float = 0.2, block: bool = True) -> Optional[FramePacket]:
        with self._cond:
            if not self._deque:
                if not self._cond.wait(timeout=timeout):
                    return None
            if self._deque:
                packet = self._deque.popleft()
                if packet is not None:
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
    def __init__(self, target_fps: float = 30.0, fps: Optional[float] = None):
        val = fps if fps is not None else target_fps
        self.target_fps = float(val) if val and val > 0 else 30.0
        self.interval_s = 1.0 / self.target_fps

    def wait_remaining(self, start_time: float) -> float:
        elapsed = time.time() - start_time
        remaining = self.interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        return max(0.0, remaining)

    def get_wait_key_delay(self, loop_time_s: float, is_live: bool = False) -> int:
        if is_live:
            return 1
        remaining_ms = (self.interval_s - loop_time_s) * 1000.0
        return max(1, int(remaining_ms))
