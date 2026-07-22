import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

recordings_dir = os.path.join('recordings')
if not os.path.exists(recordings_dir):
    os.makedirs(recordings_dir)

def process_video(video_path: str) -> None:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video {video_path}. Make sure the file is in the project folder!")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 30
        width = 640
        height = 480

        out = cv2.VideoWriter(
            os.path.join(recordings_dir, f'{timestamp}_video_processing.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (width, height)
        )

        csv_path = os.path.join(recordings_dir, f'{timestamp}_telemetry.csv')
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Object', 'X_Coord', 'Y_Coord'])

        stream_video(cap, out, csv_file, csv_writer)

    except Exception as e:
        print(f"Error in process_video: {e}")

def stream_video(cap: cv2.VideoCapture, out: cv2.VideoWriter, csv_file, csv_writer):
    trail = []
    zone_min_x, zone_max_x = 200, 440
    zone_min_y, zone_max_y = 100, 380
    zone_pts = np.array([[zone_min_x, zone_min_y], [zone_max_x, zone_min_y], [zone_max_x, zone_max_y], [zone_min_x, zone_max_y]], np.int32)
    zone_pts = zone_pts.reshape((-1, 1, 2))

    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (640, 480))
            results = model.predict(source=frame, verbose=False)
            
            zone_color = (0, 0, 255)
            alert_triggered = False
            
            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls == 14 and conf > 0.40:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    if zone_min_x <= cx <= zone_max_x and zone_min_y <= cy <= zone_max_y:
                        alert_triggered = True

            if alert_triggered:
                print("[CRITICAL] TURBINE BRAKE ENGAGED")
                if int(time.time() * 8) % 2 == 0:
                    zone_color = (0, 255, 255)
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_pts], zone_color)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.polylines(frame, [zone_pts], True, zone_color, 2)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = model.names[cls]
                
                if cls == 14 and conf > 0.40:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    trail.append((cx, cy))
                    if len(trail) > 30:
                        trail.pop(0)
                        
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    csv_writer.writerow([current_time, name, cx, cy])

            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (0, 255, 0), 2)
            
            end_time = time.time()
            loop_time = end_time - start_time
            
            if loop_time > 0:
                fps = int(1 / loop_time)
                cv2.putText(frame, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            out.write(frame)
            cv2.imshow("Day 12 - Spatial Awareness", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error in stream_video: {e}")
        
    finally:
        cap.release()
        out.release()
        csv_file.close()
        cv2.destroyAllWindows()

def visualize_latest_data():
    list_of_files = glob.glob('recordings/*.csv')
    if not list_of_files:
        print("No CSV files found in the recordings folder.")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading telemetry from: {latest_file}")

    df = pd.read_csv(latest_file)

    if df.empty:
        print("The latest CSV file is empty. No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df['X_Coord'], df['Y_Coord'], marker='o', linestyle='-', color='red', markersize=4)

    plt.title('Automated Target Trajectory Tracking')
    plt.xlabel('Camera X Coordinate (Pixels)')
    plt.ylabel('Camera Y Coordinate (Pixels)')

    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.gca().invert_yaxis() 

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    process_video("bird.mp4")
    visualize_latest_data()

if __name__ == "__main__":
    main()