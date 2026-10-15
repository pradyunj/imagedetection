# Altamont Pass Project

Wind turbines at Altamont Pass kill golden eagles and other raptors every year. This project is an autonomous computer vision and physics curtailment system that spots raptors, tracks their trajectory in real-time, differentiates protected eagles from non-protected species, and triggers turbine blade feathering before a collision can happen.

## How it works

1. **Detection & Tracking**: YOLOv8 neural detector coupled with IoU Hungarian matching and multi-object trackers (`MIL`, `CSRT`, `KCF`, `MOSSE`). Running YOLO every $N$th frame and tracking in between cuts edge latency while maintaining real-time 30+ FPS.
2. **Taxonomic Filtering**: Differentiates Class 0 (Protected Golden Eagles) from sympathetic species like Turkey Vultures (Class 1) and Common Ravens (Class 2), preventing unnecessary turbine shutdowns and power loss.
3. **4-State Kalman Filter**: Filters $[\mathbf{x}, \mathbf{y}, \mathbf{v_x}, \mathbf{v_y}]^T$ state vectors via `cv2.KalmanFilter(4, 2)` to remove wing-flap jitter and compute stable 2D velocity projections.
4. **Flight Path Tortuosity Gate & 2D Ray-Circle TTC**: Calculates path tortuosity ($\tau_{\text{tortuosity}}$) over a 30-frame window and solves 2D ray-circle intersection with the rotor swept zone (`ROTOR_RADIUS_PX = 250`). If an eagle is circling in a thermal ($\tau < 0.45$) or flying away from the rotor, curtailment alarms are suppressed. If it is on a direct approach ($\tau \ge 0.45$) with $\text{TTC} \le 15\text{ s}$, curtailment is triggered.
5. **Turbine Blade Pitch Dynamics**: Models collective blade pitch ODE ($\tau \dot{\theta} + \theta = \theta_{\text{target}}$) with rate limit $\dot{\theta}_{\text{max}} = 8^\circ/\text{s}$ and aerodynamic rotor deceleration over a 15-second Time-to-Collision (TTC) horizon.
6. **Hardware-in-the-Loop SCADA**: Communicates with turbine pitch microcontrollers (Arduino / ESP32) over USB Serial with real-time RPM simulation fallback.

## Project Structure

All runtime modules (configuration, frame buffers, detector/tracker, Kalman kinematics, SCADA serial controller, and HUD dashboard) are consolidated into `main.py`, and all plotting, ablation benchmarks, and statistical evaluations are unified in `plot_data.py`:

```
.
├── main.py               # Unified detector, tracker, Kalman kinematics, SCADA & HUD
├── plot_data.py          # Telemetry visualizer, ablation benchmark & statistical plots
├── models/
│   └── yolov8n-p2.yaml   # High-resolution stride-4 (160x160) P2 architecture spec
├── experiments/
│   ├── train_colab.ipynb # Google Colab GPU training notebook
│   └── results/          # CSV telemetry logs, reports & 300 DPI figures
├── eagle_flying.mp4      # Sample test video
└── requirements.txt      # Python dependencies
```

## How to run it

Activate your virtual environment and install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the live monitor:

```bash
# Run with default video (eagle_flying.mp4) and log telemetry CSV
python main.py

# Run on live webcam
python main.py webcam

# Run on a custom video with custom tracker and modulo settings
python main.py path/to/video.mp4 --buffer ring --modulo 5 --tracker MIL
```

Press `q` to quit. Live flight telemetry is logged automatically to `experiments/results/telemetry_latest.csv`.

### Run telemetry plotting & ablation suite:

```bash
# Run ablation benchmark, pitch brake simulation, and generate 300 DPI statistical figures
python plot_data.py

# Plot blade pitch, rotor RPM, and tortuosity time series from a telemetry CSV log
python plot_data.py experiments/results/telemetry_latest.csv
```

## Video Credit

`eagle_flying.mp4` is [Kite eagle flying](https://commons.wikimedia.org/wiki/File:Kite_eagle_flying.ogv) from Wikimedia Commons.

