# Traffic Flow Analysis Using CCTV Footage

A comprehensive computer vision and data science project for analyzing traffic patterns from CCTV footage.

## Features

- **Vehicle Detection**: Real-time vehicle detection using YOLO
- **Vehicle Classification**: Classify vehicles into categories (car, bike, bus, truck)
- **Motion Tracking**: Track vehicle movement across frames
- **Congestion Analysis**: Estimate traffic density and congestion levels
- **Data Analytics**: Statistical analysis and visualization of traffic patterns
- **Real-time Processing**: Process live CCTV feeds or recorded videos

## Project Structure

```
├── src/
│   ├── detection/          # Vehicle detection modules
│   ├── tracking/           # Object tracking algorithms
│   ├── classification/     # Vehicle classification
│   ├── analytics/          # Traffic analytics and congestion analysis
│   ├── utils/              # Utility functions
│   └── visualization/      # Data visualization tools
├── models/                 # Pre-trained models
├── data/                   # Sample videos and datasets
├── config/                 # Configuration files
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests
└── requirements.txt        # Dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download pre-trained models (see models/README.md)

## Usage

### Basic Usage

```python
from src.traffic_analyzer import TrafficAnalyzer

analyzer = TrafficAnalyzer()
analyzer.process_video("path/to/video.mp4")
```

### Real-time Analysis

```python
analyzer.process_live_feed(camera_id=0)
```

## Technologies Used

- **Computer Vision**: OpenCV, YOLO, Deep Learning
- **Tracking**: DeepSORT, Kalman Filters
- **Data Science**: Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning**: TensorFlow/PyTorch for custom models
- **Visualization**: Plotly, Matplotlib

## Applications

- Smart city traffic management
- Traffic congestion monitoring
- Vehicle counting and classification
- Traffic pattern analysis
- Urban planning insights

## License

MIT License
