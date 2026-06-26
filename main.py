import cv2
import numpy as np
from datetime import datetime
import os

recordings_dir = os.path.join('recordings')
if not os.path.exists(recordings_dir):
    os.makedirs(recordings_dir)

def record_over_webcam(channel: int) -> None:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cap = cv2.VideoCapture(channel, cv2.CAP_AVFOUNDATION)

        if not cap.isOpened():
            print(f"Error opening camera {channel}. Check macOS Privacy settings!")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 30
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            os.path.join(recordings_dir, f'{timestamp}_{channel}_webcam_recording.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (width, height)
        )

        stream_video(cap, out)

    except Exception as e:
        print(f"Error in record_over_webcam: {e}")

def stream_video(cap: cv2.VideoCapture, out: cv2.VideoWriter):
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            red_frame = frame.copy()
            red_frame[:, :, 0] = 0
            red_frame[:, :, 1] = 0
            
            dashboard = cv2.hconcat([frame, red_frame])
                
            out.write(dashboard)
            cv2.imshow("Day 3 Dashboard - Original vs Red Channel", dashboard)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error in stream_video: {e}")
        
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    record_over_webcam(0)

if __name__ == "__main__":
    main()

