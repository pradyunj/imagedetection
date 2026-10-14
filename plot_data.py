# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 13: Data Visualization (The Analyst - Standalone Telemetry Plotter)
# ==============================================================================

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_flight_telemetry(csv_path=None):
    recordings = glob.glob("experiments/results/*_telemetry.csv") or glob.glob("recordings/*_telemetry.csv") or glob.glob("*.csv")
    if csv_path is None or not os.path.exists(csv_path):
        if not recordings:
            print("[NOTICE] No flight telemetry CSV found. Run main.py first to generate telemetry.")
            return
        csv_path = max(recordings, key=os.path.getctime)

    print(f"[SYSTEM] Ingesting flight telemetry from: {csv_path}")
    df = pd.read_csv(csv_path)

    x_col = "Smooth_X" if "Smooth_X" in df.columns else ("X_Coord" if "X_Coord" in df.columns else None)
    y_col = "Smooth_Y" if "Smooth_Y" in df.columns else ("Y_Coord" if "Y_Coord" in df.columns else None)

    if df.empty or x_col is None or y_col is None:
        print("[ERROR] CSV is empty or missing coordinate columns.")
        return

    # Filter jitter and compute aerodynamic trajectory curve
    df['X_Smooth'] = df[x_col].rolling(window=7, min_periods=1, center=True).mean()
    df['Y_Smooth'] = df[y_col].rolling(window=7, min_periods=1, center=True).mean()
    sample_df = df.iloc[::4]

    plt.figure(figsize=(10, 6))

    # Plot smoothed raptor flight trajectory
    plt.plot(df['X_Smooth'], df['Y_Smooth'], linestyle='-', color='crimson', linewidth=2.5,
             label='Raptor Trajectory')
    plt.scatter(sample_df['X_Smooth'], sample_df['Y_Smooth'], color='darkred', s=22, zorder=3,
                label='Sampled Waypoints')

    # Wind turbine rotor hazard zone: [X: 200 to 440, Y: 100 to 380]
    hazard_zone = plt.Rectangle((200, 100), 240, 280, color='royalblue', alpha=0.25,
                                label='Turbine Rotor Airspace Hazard')
    plt.gca().add_patch(hazard_zone)

    plt.title('Altamont Pass Wind Farm: Autonomous Raptor Trajectory Telemetry', fontsize=13, fontweight='bold')
    plt.xlabel('Camera Sensor X Coordinate (Pixels)', fontsize=11)
    plt.ylabel('Camera Sensor Y Coordinate (Pixels)', fontsize=11)
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.gca().invert_yaxis()  # Camera coordinate system (0,0 top-left)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    output_plot = "flight_trajectory_analysis.png"
    plt.savefig(output_plot, dpi=200)
    print(f"[SYSTEM] Flight trajectory plot saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    plot_flight_telemetry()
