import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import os
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
    trail = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.predict(source=frame, verbose=False)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = model.names[cls]
                
                if cls == 67 and conf > 0.50:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    trail.append((cx, cy))
                    if len(trail) > 30:
                        trail.pop(0)
                        
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (255, 0, 0), 2)
            
            out.write(frame)
            cv2.imshow("Day 7 - Motion Tracking", frame)
            
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
 
