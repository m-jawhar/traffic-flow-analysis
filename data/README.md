# Sample Traffic Data

This directory contains sample traffic videos and datasets for testing the traffic flow analysis system.

## Sample Videos

Place your CCTV traffic footage in this directory:

- `sample_traffic.mp4` - Main sample video
- `highway_traffic.mp4` - Highway traffic footage
- `city_intersection.mp4` - City intersection footage
- `parking_lot.mp4` - Parking lot monitoring

## Download Sample Datasets

You can use the following public traffic datasets:

1. **UA-DETRAC**

   - Website: https://detrac-db.rit.albany.edu/
   - Description: 10 hours of videos captured at 24 locations in Beijing and Tianjin, China

2. **KITTI Vision Benchmark**

   - Website: http://www.cvlibs.net/datasets/kitti/
   - Description: Real-world traffic scenes with vehicles

3. **BDD100K**

   - Website: https://www.bdd100k.com/
   - Description: Diverse driving video dataset

4. **CityFlow**
   - Website: https://www.aicitychallenge.org/
   - Description: City-scale multi-camera vehicle tracking dataset

## Creating Synthetic Data

For testing purposes, you can generate synthetic traffic videos:

```python
import cv2
import numpy as np

# Create synthetic traffic video
# See notebooks/create_synthetic_data.ipynb for examples
```

## Directory Structure

```
data/
├── videos/           # Video files
├── images/           # Image frames
├── annotations/      # Ground truth annotations (optional)
└── results/          # Analysis results
```

## Video Format Requirements

- **Format**: MP4, AVI, or MOV
- **Resolution**: Minimum 640x480 (1280x720 or higher recommended)
- **Frame Rate**: 15-30 FPS
- **Duration**: Any length (system processes frame by frame)

## Quick Start

1. Place your video file in this directory
2. Update the path in the notebook or main script
3. Run the analysis

Example:

```python
from src.traffic_analyzer import TrafficFlowAnalyzer

analyzer = TrafficFlowAnalyzer()
results = analyzer.process_video("data/videos/sample_traffic.mp4")
```
