# Traffic Flow Analysis - Quick Start Guide

Welcome to the Traffic Flow Analysis system! This guide will help you get started quickly.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models (Optional)

The system will automatically download YOLOv8 when first used. If you prefer to download manually:

```bash
# YOLOv8 will auto-download on first use
# Or download from: https://github.com/ultralytics/ultralytics
```

### 3. Prepare Your Data

Place your traffic video in the `data/videos/` directory:

```
data/videos/your_traffic_video.mp4
```

## 📝 Usage Examples

### Example 1: Jupyter Notebook (Recommended for Beginners)

The easiest way to get started:

```bash
jupyter notebook notebooks/traffic_analysis_demo.ipynb
```

This notebook includes:

- Step-by-step explanations
- Visual outputs
- Complete workflow demonstration

### Example 2: Python Script

```python
from src.traffic_analyzer import TrafficFlowAnalyzer

# Initialize analyzer
analyzer = TrafficFlowAnalyzer()

# Process video
results = analyzer.process_video(
    video_source="data/videos/sample_traffic.mp4",
    output_video_path="output/videos/analyzed.mp4"
)

# View results
print(f"Total vehicles: {results['traffic_statistics']['total_vehicles_detected']}")
```

### Example 3: Command Line

```bash
python src/traffic_analyzer.py --video data/videos/sample_traffic.mp4 --output output/videos/result.mp4
```

### Example 4: Live Camera Feed

```python
from src.traffic_analyzer import TrafficFlowAnalyzer

analyzer = TrafficFlowAnalyzer()
analyzer.process_live_feed(camera_id=0)  # Use default camera
```

## 🎯 Key Features

### 1. Vehicle Detection

- Automatically detects vehicles in video frames
- Supports multiple vehicle types (car, motorcycle, bus, truck, bicycle)
- Uses state-of-the-art YOLO model

### 2. Vehicle Tracking

- Maintains consistent vehicle IDs across frames
- Tracks vehicle trajectories
- Calculates vehicle speeds

### 3. Classification

- Enhanced vehicle classification
- Feature-based analysis
- ML-based refinement

### 4. Traffic Analytics

- Vehicle counting
- Congestion level estimation
- Traffic density calculation
- Flow rate analysis

### 5. Visualization

- Real-time overlay
- Traffic heatmaps
- Interactive dashboards
- Comprehensive reports

## 📊 Output Files

After processing, you'll find:

- **Videos**: `output/videos/` - Annotated video with analysis
- **Analytics**: `output/analytics/` - JSON files with detailed statistics
- **Images**: `output/images/` - Visualization plots and charts
- **Logs**: `output/logs/` - Processing logs

## 🎨 Customization

### Modify Detection Settings

Edit `config/config.yaml`:

```yaml
detection:
  confidence_threshold: 0.5 # Adjust detection sensitivity
  model_path: "models/yolov8n.pt" # Change model
```

### Define Counting Zones

```yaml
analytics:
  counting_zones:
    - name: "Zone 1"
      polygon: [[100, 200], [300, 200], [300, 400], [100, 400]]
```

### Adjust Congestion Thresholds

```yaml
analytics:
  congestion_thresholds:
    low: 10
    medium: 25
    high: 50
```

## 🔧 Troubleshooting

### Issue: Video file not found

**Solution**: Verify the video path is correct and the file exists

### Issue: YOLO model not loading

**Solution**: Ensure internet connection for auto-download, or manually download model

### Issue: Slow processing

**Solution**: Use smaller YOLO model (yolov8n.pt) or reduce video resolution

### Issue: Poor detection accuracy

**Solution**: Increase confidence threshold or use larger YOLO model

## 📚 Learn More

- **Full Documentation**: See README.md
- **Example Notebooks**: Browse `notebooks/` directory
- **Sample Code**: Check `examples/` directory
- **API Reference**: See docstrings in source code

## 🎓 Typical Workflow

1. **Data Collection**: Gather CCTV footage
2. **Preprocessing**: Verify video quality and format
3. **Configuration**: Adjust settings in config.yaml
4. **Analysis**: Run the analyzer on your video
5. **Review**: Check output videos and analytics
6. **Refinement**: Adjust parameters as needed
7. **Deployment**: Integrate into your system

## 💡 Tips for Best Results

- Use high-resolution videos (720p or higher)
- Ensure good lighting conditions in footage
- Position camera to capture clear vehicle views
- Process shorter clips first to test settings
- Use GPU acceleration for faster processing (if available)

## 🆘 Need Help?

- Check the example notebook: `notebooks/traffic_analysis_demo.ipynb`
- Review the full README.md documentation
- Examine the example scripts in `examples/`

## 🎉 What's Next?

After getting comfortable with the basics:

1. Try different YOLO models for accuracy/speed tradeoff
2. Define custom counting zones for your specific use case
3. Integrate with databases for long-term analytics
4. Deploy on edge devices for real-time monitoring
5. Create custom reports and dashboards
6. Implement alerts for congestion events

Happy analyzing! 🚗📊
