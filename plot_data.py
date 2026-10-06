# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 13: Data Visualization (The Analyst - Standalone Telemetry Plotter)
# ==============================================================================

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_flight_telemetry(csv_path=None):
    # If no file specified, search for the latest file in recordings/ or root
    if csv_path is None or not os.path.exists(csv_path):
        recordings = glob.glob("recordings/*_telemetry.csv") or glob.glob("*.csv")
        if not recordings:
            print("[NOTICE] No flight telemetry CSV found. Generating synthetic flight path...")
            import numpy as np
            t = np.linspace(0, 10, 60)
            x = 80 + 45 * t + 12 * np.sin(t * 1.5)
            y = 380 - 28 * t + 15 * np.cos(t * 1.5)
            df = pd.DataFrame({
                "Timestamp": [f"2026-10-05 12:00:{int(i):02d}" for i in t],
                "Object": ["Golden Eagle"] * len(t),
                "X_Coord": x.astype(int),
                "Y_Coord": y.astype(int)
            })
            csv_path = "flight_data.csv"
            df.to_csv(csv_path, index=False)
        else:
            csv_path = max(recordings, key=os.path.getctime)

    print(f"[SYSTEM] Ingesting flight telemetry from: {csv_path}")
    df = pd.read_csv(csv_path)

    if df.empty or "X_Coord" not in df.columns:
        print("[ERROR] CSV is empty or missing coordinate columns.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot bird flight trajectory
    plt.plot(df['X_Coord'], df['Y_Coord'], marker='o', linestyle='-', color='crimson',
             markersize=4, label='Raptor Flight Path')

    # Draw simulated wind turbine hazard zone
    # [X: 200 to 440, Y: 100 to 380]
    hazard_zone = plt.Rectangle((200, 100), 240, 280, color='royalblue', alpha=0.25,
                                label='Turbine Rotor Airspace Hazard')
    plt.gca().add_patch(hazard_zone)

    plt.title('Altamont Pass Wind Farm: Autonomous Raptor Trajectory Telemetry', fontsize=14, fontweight='bold')
    plt.xlabel('Camera Sensor X Coordinate (Pixels)', fontsize=11)
    plt.ylabel('Camera Sensor Y Coordinate (Pixels)', fontsize=11)
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.gca().invert_yaxis()  # Match standard image coordinate system (0,0 at top-left)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    output_plot = "flight_trajectory_analysis.png"
    plt.savefig(output_plot, dpi=200)
    print(f"[SYSTEM] High-resolution science fair plot saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    plot_flight_telemetry()
