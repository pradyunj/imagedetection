# Day 14: Custom Raptor Dataset Bridge (Roboflow Universe)

## Objective
To build an elite wind turbine defense system, we transition from generic YOLO COCO classes (which only recognize generic "Bird") to custom Golden Eagle / Raptor classification.

## Phase 1: Account Creation
- Register on [Roboflow Universe](https://universe.roboflow.com).

## Phase 2: Dataset Sourcing
- Search for: `Golden Eagle Detection` or `Raptor Flight Dataset`.
- Select a dataset containing annotated bounding boxes for Golden Eagles in flight.

## Phase 3: Export Format
- Select **YOLOv8 PyTorch** format.
- Download the `data.yaml` configuration and image splits (`train/`, `val/`, `test/`).
