# Altamont Pass Project

Wind turbines at Altamont Pass kill golden eagles and other raptors every year. This project is an autonomous computer vision and physics curtailment system that spots raptors, tracks their trajectory in real-time, differentiates protected eagles from non-protected species, and triggers turbine blade feathering before a collision can happen.

## How it works

1. **Detection & Tracking**: YOLOv8 neural detector coupled with IoU Hungarian matching and multi-object trackers (MIL, CSRT, KCF, MOSSE). Running YOLO every $N$th frame and tracking in between cuts edge latency while maintaining real-time 30+ FPS.
2. **Taxonomic Filtering**: Differentiates Class 0 (Protected Eagles) from sympatric species like Turkey Vultures (Class 1) and Common Ravens (Class 2), preventing unnecessary turbine shutdowns and power loss.
3. **4-State Kalman Filter**: Filters $[x, y, v_x, v_y]^T$ state vectors to remove wing-flap jitter and compute stable velocity projections.
4. **Flight Path Tortuosity Gate**: Calculates path tortuosity ($S_{\text{tortuosity}}$) over a 30-frame window. If an eagle is circling in a thermal ($S < 0.45$), curtailment alarms are suppressed. If it is direct gliding or stooping ($S \ge 0.45$), predictive collision checks are engaged.
5. **Turbine Blade Pitch Dynamics**: Models collective blade pitch ODE ($\tau \dot{\beta} + \beta = \beta_{\text{target}}$) with rate limit $\dot{\beta}_{\max} = 8^\circ/\text{s}$ and aerodynamic rotor deceleration over a 15-second Time-to-Collision (TTC) horizon.
6. **Hardware-in-the-Loop SCADA**: Communicates with turbine pitch microcontrollers (Arduino / ESP32) over USB Serial with real-time tachometer RPM readback.

## How to run it

Activate your virtual environment and install the packages:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the live monitor:

```bash
# Run with default video (eagle_flying.mp4)
python main.py

# Run on live webcam
python main.py webcam

# Run on a custom video
python main.py path/to/video.mp4

# Advanced options:
python main.py --buffer ring --modulo 5 --tracker MIL
```

Press `q` to quit. Live video recordings and flight telemetry CSV logs are saved in `recordings/` and `experiments/results/`.

### Run experimental benchmarks and statistical suite:

```bash
# Run headless tracker and modulo ablation benchmark
python experiments/run_ablations.py

# Run ANOVA, Tukey HSD, McNemar tests, and generate 300 DPI publication plots
python experiments/stats_analysis.py

# Standalone trajectory visualizer
python plot_data.py
```

## Project Structure

```
.
├── config.py                   # Central turbine constants, zone polygons, and taxonomy
├── main.py                     # Main orchestrator
├── core/
│   ├── buffer.py               # Concurrency buffers (FIFO / Ring) and frame pacer
│   ├── detector.py             # Multi-class detector, Hungarian matching, and trackers
│   └── kinematics.py           # Kalman filter, tortuosity gate, and turbine pitch ODE
├── hardware/
│   └── scada_serial.py         # Hardware SCADA serial controller with software mock
├── ui/
│   └── dashboard.py            # Telemetry HUD with pitch gauge, RPM, and tortuosity
├── models/
│   └── yolov8n-p2.yaml         # High-resolution stride-4 (160x160) architecture
├── experiments/
│   ├── run_ablations.py        # Headless ablation benchmark runner
│   ├── stats_analysis.py       # One-way ANOVA, Tukey HSD, McNemar test, and plots
│   ├── train_cloud.py          # 4-class cloud / cluster training script
│   ├── train_colab.ipynb       # Google Colab GPU training notebook
│   └── results/                # CSV logs, reports, and 300 DPI publication figures
└── plot_data.py                # Standalone trajectory plotter
```

## Video Credit

`eagle_flying.mp4` is [Kite eagle flying](https://commons.wikimedia.org/wiki/File:Kite_eagle_flying.mp4) from Wikimedia Commons.
