"""
Simple example script demonstrating traffic flow analysis
"""

import cv2
from src.traffic_analyzer import TrafficFlowAnalyzer


def main():
    """Main function"""
    print("🚗 Traffic Flow Analysis System")
    print("=" * 50)

    # Initialize analyzer
    print("\n📊 Initializing traffic analyzer...")
    analyzer = TrafficFlowAnalyzer(config_path="config/config.yaml")

    # Option 1: Process a video file
    print("\n🎬 Processing video file...")
    video_path = "data/videos/sample_traffic.mp4"
    output_path = "output/videos/analyzed_traffic.mp4"

    try:
        results = analyzer.process_video(video_path, output_path)

        print("\n✅ Analysis complete!")
        print(f"\n📈 Results Summary:")
        print(
            f"   Total vehicles detected: {results['traffic_statistics']['total_vehicles_detected']}"
        )
        print(
            f"   Analysis duration: {results['traffic_statistics']['analysis_duration']:.2f} seconds"
        )
        print(
            f"   Average vehicles per frame: {results['traffic_statistics']['average_vehicles_per_frame']:.2f}"
        )

        vehicle_breakdown = results["traffic_statistics"]["vehicle_breakdown"]
        print(f"\n🚙 Vehicle Breakdown:")
        for vehicle_type, count in vehicle_breakdown.items():
            print(f"   {vehicle_type.capitalize()}: {count}")

    except FileNotFoundError:
        print(f"❌ Error: Video file not found at {video_path}")
        print("   Please place a video file in the data/videos/ directory")
        print("\n   You can use synthetic data for testing:")
        print("   See notebooks/traffic_analysis_demo.ipynb for examples")

    # Option 2: Process live camera feed
    # Uncomment to use live camera
    # print("\n📹 Processing live camera feed...")
    # analyzer.process_live_feed(camera_id=0)


if __name__ == "__main__":
    main()
