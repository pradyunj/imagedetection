# Altamont Pass Wind Farm: Autonomous Raptor Defense System
**Project: Pradyun's AI | Visual Output Driven Development (20-Day Engineering Pipeline)**

An intelligent, real-time edge computer vision and deep learning collision prevention platform built for wind turbines in the **Altamont Pass Wind Resource Area, California**. The system tracks incoming Golden Eagles and raptors, estimates their velocity vectors, forecasts trajectory intersection with turbine blade airspace, triggers automated braking mechanisms, and logs mission telemetry for ecological research.

---

## 🎯 Engineering Roadmap (Day 1 to Day 20)

### Week 1: Core Vision & Neural Inference
- **Day 1: Digital Workbench & Blank Canvas** — Project environment isolation (`venv`), Git versioning, `.gitignore`, and blank display window initialization (`cv2.imshow`, `cv2.waitKey`).
- **Day 2: Hardware Video Capture Pipeline** — Device initialization (`cv2.VideoCapture(0)`), infinite streaming loop, frame grabbing, and graceful keyboard break (`q`).
- **Day 3: Matrix Mathematics & Multi-View HUD** — BGR to Grayscale matrix conversion (`cv2.cvtColor`), surgical color channel isolation (pure Red feed), and horizontal matrix stacking (`np.hstack`).
- **Day 4: YOLO Deep Neural Network Integration** — Ultralytics YOLOv8 nano backbone loading, static target inference, and parsing raw bounding boxes, confidence values, and class IDs.
- **Day 5: Real-Time Live Bounding Box** — Live video inference loop merging Day 2 and Day 4, dynamic rectangle rendering (`cv2.rectangle`), and confidence score HUD overlays (`cv2.putText`).

### Week 2: Target Filtering, Trajectory & Telemetry
- **Day 6: Targeted Object Filtering ("Sniper" Focus)** — Class ID gating (filtering specifically for birds/raptors), confidence thresholding ($> 60\%$), and console acquisition alerts (`[ALERT] TARGET ACQUIRED!`).
- **Day 7: Motion Tracking & Trajectory Trail** — Center-point coordinate math $((x_1+x_2)/2, (y_1+y_2)/2)$, 30-frame historical coordinate queue, and continuous trajectory tail rendering (`cv2.line`).
- **Day 8: Performance Analytics (FPS Dashboard)** — System loop latency calculation (`time.perf_counter()`), frame rate calculation ($FPS = 1 / \Delta t$), and inference downscaling (`cv2.resize`).
- **Day 9: Data Logging (Researcher's Ledger)** — Continuous CSV telemetry streaming (`flight_data.csv`) capturing `[Timestamp, Object, X_Coord, Y_Coord]` with atomic file flushes and clean exit closure.

### Week 3: Geofencing, Collision Triggers & Science Fair Analytics
- **Day 10: Local Field Video Asset Processing** — Dynamic video asset ingestion (`.mp4` field drone/wildlife clips) with graceful loop termination on final frame.
- **Day 11: Turbine Airspace Danger Polygon Geofencing** — Mapping turbine rotor airspace hazard polygon and rendering semi-transparent danger zones using alpha blending (`cv2.addWeighted`).
- **Day 12: Collision Trigger & Turbine Braking** — Spatial intersection testing between raptor coordinates and hazard boundaries, visual yellow strobe alarms, and terminal brake triggers (`[CRITICAL] TURBINE BRAKE ENGAGED`).
- **Day 13: Data Science Visualization (Matplotlib & Pandas)** — Standalone analysis pipeline (`plot_data.py`) ingesting flight logs and plotting 2D trajectory paths against turbine airspace for science fair presentations.

### Week 4: Cloud Training, Multi-Threading & Mission Control UI
- **Day 14: Custom Raptor Dataset Bridge** — Sourcing and curating Roboflow Golden Eagle datasets in YOLOv8 PyTorch format.
- **Day 15: Cloud GPU Accelerated Training** — Google Colab / cloud GPU training pipeline for 50 epochs exporting fine-tuned weights (`best.pt`).
- **Day 16: Producer-Consumer Multi-Threading** — Decoupling video I/O from heavy neural inference using `threading.Thread` and `queue.Queue` to eliminate video stuttering.
- **Day 17: Predictive Trajectory & Velocity Vectors** — Calculating instantaneous velocity vectors $(v_x, v_y)$, projecting 30 frames forward, and performing line-clipping predictive collision checks (`cv2.clipLine`).
- **Day 18: Edge Hardware Optimization (Frame Dropping)** — Modulo inference scheduling (`frame_count % 5 == 0`) combined with lightweight KCF pixel tracking for intermediate frames (60+ FPS).
- **Day 19: Professional Object-Oriented Architecture** — Clean encapsulation within `RaptorDefenseSystem` class with modular methods (`check_collision`, `write_telemetry`, `render_dashboard`).
- **Day 20: Command Center Industrial Mission Control UI** — Integrated 900x480 dashboard featuring a dark telemetry sidebar, live coordinates, FPS counter, and an industrial status indicator button (Green "Airspace Secure" / Flashing Red "Brake Engaged").

---

## 📁 Repository Structure

```text
Altamont Pass Project/
├── main.py                     # Unified Day 20 Production Command Center
├── plot_data.py                # Standalone Day 13 Flight Trajectory Plotter
├── requirements.txt            # Locked package dependencies
├── .gitignore                  # Git tracking rules
├── recordings/                 # Automated output folder (MP4 recordings & CSV telemetry)
└── milestones/                 # Individual step-by-step scripts for each day
    ├── day01_blank_canvas.py
    ├── day02_webcam_stream.py
    ├── day03_color_manipulation.py
    ├── day04_yolo_static.py
    ├── day05_live_bounding_box.py
    ├── day06_target_filter.py
    ├── day07_motion_trail.py
    ├── day08_fps_dashboard.py
    ├── day09_telemetry_logger.py
    ├── day10_video_file.py
    ├── day11_danger_zone.py
    ├── day12_turbine_brake.py
    ├── day13_data_visualization.py
    ├── day14_dataset_prep.md
    ├── day15_cloud_training.py
    ├── day16_multithreading.py
    ├── day17_trajectory_prediction.py
    ├── day18_edge_optimization.py
    ├── day19_oop_architecture.py
    └── day20_command_center.py
```

---

## 🚀 How to Run

### 1. Activate Environment
Open your terminal inside the project directory:
```bash
cd "/Users/pradyunjupalli/Documents/Altamont Pass Project"
source .venv/bin/activate
```

### 2. Run the Full Command Center (Day 20 Production System)
```bash
python main.py
```
- If a video file named `birdtest.mp4` is in the folder, it will process the video file.
- If no video file is present, it will automatically connect to your live webcam (`Index 0`).
- Press **`q`** at any time to initiate a safe shutdown, auto-save the video recording and CSV telemetry, and display the Matplotlib flight trajectory analysis!

### 3. Run Standalone Trajectory Analysis
```bash
python plot_data.py
```
Ingests the latest CSV telemetry log from `recordings/` and exports a high-resolution graph `flight_trajectory_analysis.png`.

### 4. Run Any Individual Daily Milestone
Each day from Satish's curriculum can be run independently:
```bash
python milestones/day01_blank_canvas.py
python milestones/day05_live_bounding_box.py
python milestones/day12_turbine_brake.py
python milestones/day17_trajectory_prediction.py
```
