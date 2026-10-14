import os
import sys
import time
import argparse
import csv
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from core.detector import MultiTargetDetector, compute_iou

def extract_primary_bbox(detector: MultiTargetDetector) -> Optional[Tuple[int, int, int, int]]:
    primary = detector.get_primary_target()
    return primary.bbox if primary else None

def run_single_ablation(
    video_path: str,
    modulo_n: int,
    tracker_type: str,
    baseline_boxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
    max_frames: int = 0
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    model_weights = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
    detector = MultiTargetDetector(
        model_path=model_weights,
        modulo_n=modulo_n,
        tracker_type=tracker_type,
        conf_threshold=0.35
    )

    frame_latencies_ms: List[float] = []
    detected_boxes: List[Optional[Tuple[int, int, int, int]]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        t0 = time.perf_counter()
        detector.update(frame_resized, frame_idx)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        frame_latencies_ms.append(latency_ms)

        bbox = extract_primary_bbox(detector)
        detected_boxes.append(bbox)

    cap.release()

    total_time_s = sum(frame_latencies_ms) / 1000.0
    mean_fps = frame_idx / max(total_time_s, 1e-6)
    p95_lat = float(np.percentile(frame_latencies_ms, 95))
    p99_lat = float(np.percentile(frame_latencies_ms, 99))
    mean_lat = float(np.mean(frame_latencies_ms))
    min_lat = float(np.min(frame_latencies_ms))
    max_lat = float(np.max(frame_latencies_ms))

    # Calculate IoU drift against baseline if baseline is provided
    ious: List[float] = []
    if baseline_boxes is not None:
        limit = min(len(baseline_boxes), len(detected_boxes))
        for i in range(limit):
            b_base = baseline_boxes[i]
            b_curr = detected_boxes[i]
            if b_base is not None and b_curr is not None:
                ious.append(compute_iou(b_base, b_curr))
            elif b_base is None and b_curr is None:
                ious.append(1.0)
            else:
                ious.append(0.0)

    mean_iou = float(np.mean(ious)) if ious else 1.0
    median_iou = float(np.median(ious)) if ious else 1.0

    return {
        "tracker": tracker_type,
        "modulo_n": modulo_n,
        "total_frames": frame_idx,
        "mean_fps": round(mean_fps, 2),
        "min_latency_ms": round(min_lat, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "mean_iou": round(mean_iou, 4),
        "median_iou": round(median_iou, 4),
        "prevented_curtailments": detector.prevented_curtailments_count,
        "boxes": detected_boxes
    }

def run_suite(
    video_path: str = config.DEFAULT_VIDEO_SOURCE,
    output_csv: str = "experiments/results/ablation_results.csv",
    max_frames: int = 150
):
    print("=" * 78)
    print("ISEF BENCHMARK: EDGE CONCURRENCY & TRACKER MODULO ABLATION SUITE")
    print(f"Target Video: {video_path} | Max Frames/Trial: {max_frames if max_frames > 0 else 'Full'}")
    print("=" * 78)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 1. Establish N=1 Baseline Ground Truth
    print("[BASELINE] Running N=1 pure neural inference baseline...")
    baseline_run = run_single_ablation(
        video_path=video_path,
        modulo_n=1,
        tracker_type="MIL",
        baseline_boxes=None,
        max_frames=max_frames
    )
    baseline_boxes = baseline_run["boxes"]
    print(f"[BASELINE] Complete. Base FPS: {baseline_run['mean_fps']} | P99 Latency: {baseline_run['p99_latency_ms']} ms\n")

    modulo_intervals = [1, 2, 3, 5, 8, 10, 15, 20]
    trackers = ["MIL", "CSRT", "KCF", "MOSSE"]

    results: List[Dict[str, Any]] = []

    for tracker in trackers:
        print(f"--- Benchmarking Tracker: {tracker} ---")
        for n in modulo_intervals:
            print(f"  [RUN] Tracker={tracker:<5} | Modulo N={n:<2} ... ", end="", flush=True)
            res = run_single_ablation(
                video_path=video_path,
                modulo_n=n,
                tracker_type=tracker,
                baseline_boxes=baseline_boxes,
                max_frames=max_frames
            )
            print(f"FPS: {res['mean_fps']:<6} | Mean IoU: {res['mean_iou']:<6} | P99 Lat: {res['p99_latency_ms']} ms")
            results.append(res)

    # Write results to CSV
    fields = [
        "tracker", "modulo_n", "total_frames", "mean_fps",
        "min_latency_ms", "mean_latency_ms", "p95_latency_ms", "p99_latency_ms", "max_latency_ms",
        "mean_iou", "median_iou", "prevented_curtailments"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\n" + "=" * 78)
    print(f"[SUMMARY] All experimental trials completed. Results saved to:\n  -> {output_csv}")
    print("=" * 78)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Multi-Tracker Modulo Ablation Benchmark")
    parser.add_argument("--video", type=str, default=config.DEFAULT_VIDEO_SOURCE, help="Input video path")
    parser.add_argument("--out", type=str, default="experiments/results/ablation_results.csv", help="Output CSV path")
    parser.add_argument("--max-frames", type=int, default=120, help="Frames per condition (0 for entire video)")
    args = parser.parse_args()

    run_suite(video_path=args.video, output_csv=args.out, max_frames=args.max_frames)
