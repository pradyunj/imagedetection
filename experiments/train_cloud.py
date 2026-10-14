import os
import argparse
import yaml
from ultralytics import YOLO

def create_dataset_config(output_path: str = "experiments/altamont_raptors.yaml"):
    config_dict = {
        "path": os.path.abspath("data"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "Eagle_Protected",
            1: "Sympatric_Vulture",
            2: "Sympatric_Corvid_Buteo",
            3: "Anthropogenic_Airborne"
        }
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"[DATASET] Wrote dataset configuration to: {output_path}")
    return output_path

def train_model(
    config_file: str,
    architecture_yaml: str = "models/yolov8n-p2.yaml",
    pretrained_weights: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",
    project_name: str = "experiments/results/train"
):
    print(f"[TRAINING] Initializing {architecture_yaml} with transfer weights from {pretrained_weights}...")
    model = YOLO(architecture_yaml).load(pretrained_weights)

    print(f"[TRAINING] Starting training across {epochs} epochs (Image size: {imgsz}, Batch: {batch_size})...")
    results = model.train(
        data=config_file,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=25,
        save=True,
        optimizer="AdamW",
        lr0=0.001,
        project=project_name,
        name="yolov8n_p2_4class",
        exist_ok=True
    )

    print("[TRAINING] Validating best checkpoint...")
    metrics = model.val()
    print(f"[TRAINING] Validation mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")

    export_path = model.export(format="onnx", dynamic=True, simplify=True)
    print(f"[TRAINING] Exported optimized model to: {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4-Class YOLOv8n-P2 Raptor Model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--arch", type=str, default="models/yolov8n-p2.yaml")
    args = parser.parse_args()

    cfg = create_dataset_config()
    train_model(
        config_file=cfg,
        architecture_yaml=args.arch,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device
    )
