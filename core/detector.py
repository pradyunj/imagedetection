import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

import config

@dataclass
class TrackedTarget:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    tracker: Any = None
    missed_frames: int = 0
    is_active: bool = True

def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

def create_tracker_instance(tracker_type: str = "MIL"):
    tt = tracker_type.upper()
    if tt == "KCF":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
    elif tt == "CSRT":
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
    elif tt == "MOSSE":
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
    
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
        return cv2.legacy.TrackerMIL_create()
    return None

class MultiTargetDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_type: str = config.DEFAULT_TRACKER,
        modulo_n: int = config.DEFAULT_MODULO_N,
        iou_match_threshold: float = 0.25,
        max_missed_frames: int = 5
    ):
        actual_model = "best.pt" if os.path.exists("best.pt") else model_path
        self.model = YOLO(actual_model)
        self.is_custom_model = os.path.exists("best.pt")
        self.tracker_type = tracker_type
        self.modulo_n = modulo_n
        self.iou_match_threshold = iou_match_threshold
        self.max_missed_frames = max_missed_frames
        
        self.targets: Dict[int, TrackedTarget] = {}
        self.next_track_id = 1
        self.prevented_curtailments_count = 0

    def _map_detection_class(self, raw_cls: int, conf: float) -> Tuple[int, str]:
        if self.is_custom_model:
            cls_id = int(raw_cls)
            name = config.TAXONOMIC_NAMES.get(cls_id, "Unknown Target")
            return cls_id, name

        raw_name = self.model.names.get(raw_cls, "").lower()
        if "bird" in raw_name or raw_cls == 14:
            return config.CLASS_EAGLE_PROTECTED, "Eagle (Protected)"
        elif "airplane" in raw_name or raw_cls == 4:
            return config.CLASS_ANTHROPOGENIC_AIRBORNE, "Aircraft / Drone"
        elif raw_cls == 0:
            return config.CLASS_EAGLE_PROTECTED, "Target"
        return config.CLASS_SYMPATRIC_CORVID_BUTEO, "Sympatric Non-Eagle"

    def _yolo_detection_pass(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], int, str, float]]:
        results = self.model.predict(source=frame, verbose=False)
        detections = []
        if not results or not len(results[0].boxes):
            return detections

        for box in results[0].boxes:
            conf = float(box.conf[0].item())
            if conf < 0.35:
                continue

            raw_cls = int(box.cls[0].item())
            class_id, class_name = self._map_detection_class(raw_cls, conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            detections.append((bbox, class_id, class_name, conf))
        return detections

    def _associate_and_update(self, frame: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], int, str, float]]):
        existing_ids = list(self.targets.keys())
        if not existing_ids:
            for bbox, cid, cname, conf in detections:
                t = create_tracker_instance(self.tracker_type)
                if t:
                    t.init(frame, bbox)
                cx = bbox[0] + bbox[2] // 2
                cy = bbox[1] + bbox[3] // 2
                self.targets[self.next_track_id] = TrackedTarget(
                    track_id=self.next_track_id,
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                    bbox=bbox,
                    center=(cx, cy),
                    tracker=t
                )
                self.next_track_id += 1
            return

        if not detections:
            for tid in existing_ids:
                self.targets[tid].missed_frames += 1
                if self.targets[tid].missed_frames > self.max_missed_frames:
                    del self.targets[tid]
            return

        cost_matrix = np.zeros((len(existing_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(existing_ids):
            for j, (det_bbox, _, _, _) in enumerate(detections):
                iou = compute_iou(self.targets[tid].bbox, det_bbox)
                cost_matrix[i, j] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            iou = 1.0 - cost_matrix[r, c]
            if iou >= self.iou_match_threshold:
                tid = existing_ids[r]
                det_bbox, cid, cname, conf = detections[c]
                t = create_tracker_instance(self.tracker_type)
                if t:
                    t.init(frame, det_bbox)
                cx = det_bbox[0] + det_bbox[2] // 2
                cy = det_bbox[1] + det_bbox[3] // 2
                
                target = self.targets[tid]
                target.bbox = det_bbox
                target.center = (cx, cy)
                target.class_id = cid
                target.class_name = cname
                target.confidence = conf
                target.tracker = t
                target.missed_frames = 0
                matched_tracks.add(r)
                matched_dets.add(c)

        for i, tid in enumerate(existing_ids):
            if i not in matched_tracks:
                self.targets[tid].missed_frames += 1
                if self.targets[tid].missed_frames > self.max_missed_frames:
                    del self.targets[tid]

        for j, (det_bbox, cid, cname, conf) in enumerate(detections):
            if j not in matched_dets:
                t = create_tracker_instance(self.tracker_type)
                if t:
                    t.init(frame, det_bbox)
                cx = det_bbox[0] + det_bbox[2] // 2
                cy = det_bbox[1] + det_bbox[3] // 2
                self.targets[self.next_track_id] = TrackedTarget(
                    track_id=self.next_track_id,
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                    bbox=det_bbox,
                    center=(cx, cy),
                    tracker=t
                )
                self.next_track_id += 1

    def _tracker_update_pass(self, frame: np.ndarray):
        to_remove = []
        for tid, target in self.targets.items():
            if target.tracker is None:
                continue
            success, bbox = target.tracker.update(frame)
            if success:
                target.bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                target.center = (target.bbox[0] + target.bbox[2] // 2, target.bbox[1] + target.bbox[3] // 2)
                target.missed_frames = 0
            else:
                target.missed_frames += 1
                if target.missed_frames > self.max_missed_frames:
                    to_remove.append(tid)

        for tid in to_remove:
            del self.targets[tid]

    def update(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[TrackedTarget], str]:
        run_yolo = (frame_idx % self.modulo_n == 0) or (len(self.targets) == 0)
        mode = "YOLO INFERENCE" if run_yolo else f"{self.tracker_type} TRACKER"

        if run_yolo:
            detections = self._yolo_detection_pass(frame)
            self._associate_and_update(frame, detections)
        else:
            self._tracker_update_pass(frame)

        # Check for non-protected species to record prevented false alarms
        for target in self.targets.values():
            if target.class_id != config.CLASS_EAGLE_PROTECTED:
                self.prevented_curtailments_count += 1

        active = [t for t in self.targets.values() if t.missed_frames == 0]
        return active, mode

    def get_primary_target(self) -> Optional[TrackedTarget]:
        if not self.targets:
            return None
        protected = [t for t in self.targets.values() if t.class_id == config.CLASS_EAGLE_PROTECTED and t.missed_frames == 0]
        if protected:
            return max(protected, key=lambda t: t.confidence)
        active = [t for t in self.targets.values() if t.missed_frames == 0]
        if active:
            return max(active, key=lambda t: t.confidence)
        return None
