#!/usr/bin/env python3
"""
Demo script that uses synthetic data instead of requiring video files
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_analyzer import TrafficFlowAnalyzer


def create_synthetic_frame(
    frame_num: int, width: int = 1920, height: int = 1080
) -> np.ndarray:
    """Create a synthetic frame with moving vehicles"""
    # Create blank frame
    frame = np.ones((height, width, 3), dtype=np.uint8) * 50

    # Draw road
    cv2.rectangle(frame, (0, height // 3), (width, 2 * height // 3), (80, 80, 80), -1)

    # Add lane markings
    for y in [height // 2]:
        for x in range(0, width, 100):
            cv2.rectangle(frame, (x, y - 2), (x + 50, y + 2), (255, 255, 255), -1)

    # Simulate moving vehicles
    num_vehicles = 5 + (frame_num % 10)  # Varying traffic
    for i in range(num_vehicles):
        # Vehicle position moves over time
        x = int((100 + i * 300 + frame_num * 5) % (width + 200)) - 100
        y = int(height // 2 + np.sin(i * 0.5) * 100)

        # Draw simple vehicle (rectangle)
        vehicle_width, vehicle_height = 80, 50
        color = (
            np.random.randint(50, 200),
            np.random.randint(50, 200),
            np.random.randint(50, 200),
        )
        cv2.rectangle(frame, (x, y), (x + vehicle_width, y + vehicle_height), color, -1)

        # Add some details
        cv2.rectangle(
            frame, (x + 5, y + 5), (x + 35, y + 20), (200, 200, 255), -1
        )  # Window
        cv2.rectangle(
            frame, (x + 45, y + 5), (x + 75, y + 20), (200, 200, 255), -1
        )  # Window

    return frame


def main():
    """Main demo function"""
    print("🚗 Traffic Flow Analysis - Synthetic Data Demo")
    print("=" * 50)
    print()

    # Initialize analyzer
    print("📊 Initializing traffic analyzer...")
    config_path = project_root / "config" / "config.yaml"
    analyzer = TrafficFlowAnalyzer(str(config_path))
    print("✓ Analyzer initialized")
    print()

    # Process synthetic frames
    print("🎬 Processing synthetic video frames...")
    num_frames = 100  # Simulate 100 frames

    for frame_num in range(num_frames):
        # Create synthetic frame
        frame = create_synthetic_frame(frame_num)

        # Process frame
        processed_frame, detections, tracked_objects = analyzer.process_frame(frame)

        if frame_num % 10 == 0:
            print(
                f"Frame {frame_num}: {len(detections)} vehicles detected, "
                f"{len(tracked_objects)} tracked"
            )

    print()
    print("✓ Processing complete")
    print()

    # Get analytics
    print("📈 Traffic Analytics:")
    print("-" * 50)
    statistics = analyzer.get_traffic_statistics()

    if not statistics:
        print("No statistics available (no traffic detected)")
        print()
    else:
        print(
            f"Total vehicles detected: {statistics.get('total_vehicles_detected', 0)}"
        )
        print(
            f"Average vehicles per frame: {statistics.get('average_vehicles_per_frame', 0):.2f}"
        )
        print(f"Peak vehicle count: {statistics.get('peak_vehicle_count', 0)}")
        print()

        print("Vehicle breakdown:")
        for vehicle_type, count in statistics.get("vehicle_breakdown", {}).items():
            print(f"  {vehicle_type}: {count}")
        print()

        print("Congestion distribution:")
        for level, count in statistics.get("congestion_distribution", {}).items():
            print(f"  {level}: {count} frames")
        print()

    # Generate visualizations
    print("📊 Generating visualizations...")
    output_dir = project_root / "outputs" / "synthetic_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save analytics to JSON
        import json

        analytics_path = output_dir / "analytics.json"
        if statistics:
            with open(analytics_path, "w") as f:
                json.dump(statistics, f, indent=2, default=str)
            print(f"✓ Analytics saved to: {analytics_path}")

        # Note: Visualizations require traffic_flow_data which is available after process_video
        # For frame-by-frame processing, visualizations need to be generated differently
        print(
            "Note: For synthetic frame-by-frame processing, use process_video() for full analytics"
        )

    except Exception as e:
        print(f"⚠️ Warning: Could not save results: {e}")

    print()
    print("🎉 Demo complete!")
    print(f"📁 Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
