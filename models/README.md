# Models Directory

This directory contains pre-trained models for vehicle detection and classification.

## Required Models

### YOLO Models

Download one of the following YOLO models and place them in this directory:

1. **YOLOv8 (Recommended)**

   ```bash
   # The model will be automatically downloaded when first used
   # Or manually download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

2. **YOLOv5**
   ```bash
   # Download from: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
   ```

### Vehicle Classification Models

- `vehicle_classifier.pkl` - Trained RandomForest classifier for enhanced vehicle classification
- `scaler.pkl` - Feature scaler for the classifier

## Model Files

Place the following files here:

- `yolov8n.pt` - YOLOv8 nano (fastest)
- `yolov8s.pt` - YOLOv8 small (balanced)
- `yolov8m.pt` - YOLOv8 medium (accurate)
- `yolov8l.pt` - YOLOv8 large (most accurate)

## Usage

The detection module will automatically load models from this directory based on the configuration in `config/config.yaml`.
