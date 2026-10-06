# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 13: Data Visualization (The Analyst - Flight Path Trajectory Graph)
# ==============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# Phase 1: Check for CSV data
csv_file = "flight_data.csv"
if not os.path.exists(csv_file):
    print(f"[SYSTEM] '{csv_file}' not found. Generating sample flight path data...")
    import numpy as np
    t = np.linspace(0, 10, 50)
    x = 100 + 40 * t + 10 * np.sin(t)
    y = 350 - 25 * t + 15 * np.cos(t)
    df_sample = pd.DataFrame({
        "Timestamp": [f"2026-10-05 12:00:{int(i):02d}" for i in t],
        "Object": ["Eagle"] * len(t),
        "X_Coord": x.astype(int),
        "Y_Coord": y.astype(int)
    })
    df_sample.to_csv(csv_file, index=False)

# Phase 2: Ingest CSV into Pandas DataFrame
df = pd.read_csv(csv_file)
print(f"[SYSTEM] Ingested {len(df)} telemetry points from {csv_file}")

# Phase 3: Generate 2D Trajectory Graph
plt.figure(figsize=(10, 6))
plt.plot(df['X_Coord'], df['Y_Coord'], marker='o', linestyle='-', color='red', markersize=4, label='Flight Path')

# Turbine Danger Zone Box
plt.gca().add_patch(plt.Rectangle((200, 100), 240, 280, color='blue', alpha=0.2, label='Turbine Airspace'))

plt.title('Raptor Defense System: Flight Trajectory Telemetry')
plt.xlabel('Camera X Coordinate (Pixels)')
plt.ylabel('Camera Y Coordinate (Pixels)')
plt.xlim(0, 640)
plt.ylim(0, 480)
plt.gca().invert_yaxis()  # Match camera pixel coordinates (0 at top)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("flight_path_visualization.png")
print("[SYSTEM] Saved plot to flight_path_visualization.png")
plt.show()
print("[SYSTEM] Day 13 Complete! Trajectory plot rendered.")
