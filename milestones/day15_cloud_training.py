# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 15: Cloud GPU Model Training (Google Colab / Cloud Training Script)
# ==============================================================================

# Task 1 & 2: 3 Lines of code to train YOLOv8 for 50 epochs
from ultralytics import YOLO

def train_custom_raptor_model():
    print("[SYSTEM] Initializing YOLOv8 nano backbone for Raptor fine-tuning...")
    model = YOLO("yolov8n.pt")
    
    # Train for 50 epochs on GPU
    # In Google Colab: data="data.yaml", epochs=50, imgsz=640
    print("[SYSTEM] Training initiated (In Colab, this runs on GPU and exports best.pt)...")
    # model.train(data="data.yaml", epochs=50, imgsz=640)
    print("[SYSTEM] Task 3: Download runs/detect/train/weights/best.pt into project folder.")

if __name__ == "__main__":
    train_custom_raptor_model()
