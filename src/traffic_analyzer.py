"""
Main Traffic Flow Analyzer
Integrates all components for comprehensive traffic analysis
"""

import cv2
import numpy as np
import yaml
import os
import argparse
from datetime import datetime
from typing import Optional, Dict, List
import json

# Import project modules
from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.classification.vehicle_classifier import VehicleClassifier
from src.analytics.traffic_analyzer import TrafficAnalyzer
from src.visualization.traffic_visualizer import TrafficVisualizer
from src.utils.helpers import (
    load_config,
    create_output_directories,
    resize_frame,
    create_video_writer,
    log_message,
    validate_video_source,
    get_video_properties,
    PerformanceMonitor,
)


class TrafficFlowAnalyzer:
    """Main traffic flow analysis system"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the traffic flow analyzer

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.output_dirs = create_output_directories(
            self.config["output"]["output_dir"]
        )

        # Initialize components
        self.detector = VehicleDetector(config_path)
        self.tracker = VehicleTracker(
            tracker_type=self.config["tracking"]["tracker_type"],
            max_disappeared=self.config["tracking"]["max_disappeared"],
            max_distance=self.config["tracking"]["max_distance"],
        )
        self.classifier = VehicleClassifier()
        self.analytics = TrafficAnalyzer(self.config)
        self.visualizer = TrafficVisualizer()

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # State variables
        self.frame_count = 0
        self.is_processing = False

        # Setup logging
        self.log_file = os.path.join(
            self.output_dirs["logs"],
            f"traffic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

    def process_video(
        self, video_source, output_video_path: Optional[str] = None
    ) -> Dict:
        """
        Process video for traffic analysis

        Args:
            video_source: Video file path or camera index
            output_video_path: Path to save output video

        Returns:
            Analysis results dictionary
        """
        log_message(f"Starting traffic analysis on: {video_source}", self.log_file)

        # Validate video source
        if not validate_video_source(video_source):
            raise ValueError(f"Invalid video source: {video_source}")

        # Open video capture
        cap = cv2.VideoCapture(video_source)

        # Get video properties
        if isinstance(video_source, str):
            video_props = get_video_properties(video_source)
            log_message(f"Video properties: {video_props}", self.log_file)

        # Setup video writer if output path is specified
        video_writer = None
        if output_video_path and self.config["output"]["save_video"]:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = create_video_writer(output_video_path, fps, (width, height))

        self.is_processing = True
        all_analysis_results = []

        try:
            while cap.isOpened() and self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start_time = cv2.getTickCount()

                # Process single frame
                frame_result = self._process_frame(frame)
                all_analysis_results.append(frame_result["analysis"])

                # Draw visualizations
                output_frame = self._draw_frame_visualization(frame, frame_result)

                # Write frame to video if enabled
                if video_writer:
                    video_writer.write(output_frame)

                # Display frame (optional)
                if self.config.get("display", {}).get("show_live", False):
                    cv2.imshow("Traffic Analysis", resize_frame(output_frame, 1280))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Update performance monitoring
                frame_time = (
                    cv2.getTickCount() - frame_start_time
                ) / cv2.getTickFrequency()
                self.performance_monitor.update(frame_time)

                # Log progress
                if self.frame_count % 100 == 0:
                    fps = self.performance_monitor.get_fps()
                    log_message(
                        f"Processed {self.frame_count} frames - FPS: {fps:.1f}",
                        self.log_file,
                    )

        except KeyboardInterrupt:
            log_message("Processing interrupted by user", self.log_file)

        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

        # Generate final analysis
        final_results = self._generate_final_analysis(all_analysis_results)

        log_message(
            f"Analysis completed. Processed {self.frame_count} frames", self.log_file
        )
        return final_results

    def process_live_feed(self, camera_id: int = 0) -> None:
        """
        Process live camera feed

        Args:
            camera_id: Camera device ID
        """
        log_message(
            f"Starting live feed analysis from camera {camera_id}", self.log_file
        )

        output_video_path = None
        if self.config["output"]["save_video"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = os.path.join(
                self.output_dirs["videos"], f"live_analysis_{timestamp}.mp4"
            )

        self.process_video(camera_id, output_video_path)

    def _process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame"""
        self.frame_count += 1

        # Detect vehicles
        detections = self.detector.detect_vehicles(frame)

        # Track vehicles
        tracked_objects = self.tracker.update(detections)

        # Classify vehicles (enhanced classification)
        for obj_id, obj_info in tracked_objects.items():
            detection = obj_info["detection"]
            bbox = detection["bbox"]

            # Crop vehicle region
            x1, y1, x2, y2 = bbox
            vehicle_crop = frame[y1:y2, x1:x2]

            if vehicle_crop.size > 0:
                # Enhanced classification
                classification_result = self.classifier.classify_vehicle(
                    vehicle_crop, detection
                )
                obj_info["enhanced_classification"] = classification_result

        # Analyze traffic
        analysis_result = self.analytics.analyze_frame(tracked_objects, frame.shape)

        return {
            "frame_number": self.frame_count,
            "detections": detections,
            "tracked_objects": tracked_objects,
            "analysis": analysis_result,
        }

    def _draw_frame_visualization(
        self, frame: np.ndarray, frame_result: Dict
    ) -> np.ndarray:
        """Draw all visualizations on frame"""
        output_frame = frame.copy()

        # Draw detections if enabled
        if self.config["visualization"]["show_bboxes"]:
            output_frame = self.detector.draw_detections(
                output_frame, frame_result["detections"]
            )

        # Draw tracks if enabled
        if self.config["visualization"]["show_tracks"]:
            output_frame = self.tracker.draw_tracks(
                output_frame, frame_result["tracked_objects"]
            )

        # Draw analytics overlay
        output_frame = self.analytics.draw_analytics_overlay(
            output_frame, frame_result["analysis"]
        )

        return output_frame

    def _generate_final_analysis(self, all_analysis_results: List[Dict]) -> Dict:
        """Generate final comprehensive analysis"""
        # Get traffic statistics
        traffic_stats = self.analytics.get_traffic_statistics()

        # Get tracking statistics
        tracking_stats = self.tracker.get_track_statistics()

        # Get performance statistics
        performance_stats = {
            "avg_fps": self.performance_monitor.get_fps(),
            "avg_processing_time": self.performance_monitor.get_average_processing_time(),
            "total_frames": self.frame_count,
        }

        final_results = {
            "traffic_statistics": traffic_stats,
            "tracking_statistics": tracking_stats,
            "performance_statistics": performance_stats,
            "frame_by_frame_data": all_analysis_results,
            "analysis_timestamp": datetime.now().isoformat(),
            "configuration": self.config,
        }

        # Save results
        self._save_results(final_results, all_analysis_results)

        # Generate visualizations
        self._generate_visualizations(all_analysis_results, traffic_stats)

        return final_results

    def _save_results(self, final_results: Dict, frame_data: List[Dict]):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comprehensive results
        results_path = os.path.join(
            self.output_dirs["analytics"], f"traffic_analysis_results_{timestamp}.json"
        )
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        # Save analytics data
        analytics_path = os.path.join(
            self.output_dirs["analytics"], f"analytics_data_{timestamp}.json"
        )
        self.analytics.export_analytics(analytics_path)

        log_message(f"Results saved to {results_path}", self.log_file)
        log_message(f"Analytics data saved to {analytics_path}", self.log_file)

    def _generate_visualizations(self, frame_data: List[Dict], traffic_stats: Dict):
        """Generate visualization plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Vehicle counts over time
            counts_path = os.path.join(
                self.output_dirs["images"], f"vehicle_counts_{timestamp}.png"
            )
            self.visualizer.plot_vehicle_counts_over_time(frame_data, counts_path)

            # Congestion levels
            congestion_path = os.path.join(
                self.output_dirs["images"], f"congestion_levels_{timestamp}.png"
            )
            self.visualizer.plot_congestion_levels(frame_data, congestion_path)

            # Vehicle type distribution
            if traffic_stats.get("vehicle_breakdown"):
                distribution_path = os.path.join(
                    self.output_dirs["images"], f"vehicle_distribution_{timestamp}.png"
                )
                self.visualizer.plot_vehicle_type_distribution(
                    traffic_stats, distribution_path
                )

            # Traffic density heatmap
            heatmap_path = os.path.join(
                self.output_dirs["images"], f"traffic_heatmap_{timestamp}.png"
            )
            self.visualizer.plot_traffic_density_heatmap(frame_data, heatmap_path)

            # Generate summary report
            report_path = os.path.join(
                self.output_dirs["analytics"], f"summary_report_{timestamp}.md"
            )
            self.visualizer.generate_summary_report(traffic_stats, report_path)

            log_message(
                f"Visualizations generated in {self.output_dirs['images']}",
                self.log_file,
            )

        except Exception as e:
            log_message(f"Error generating visualizations: {str(e)}", self.log_file)

    def stop_processing(self):
        """Stop the processing loop"""
        self.is_processing = False

    def process_frame(self, frame: np.ndarray):
        """
        Process a single frame (public API)

        Args:
            frame: Input frame

        Returns:
            Tuple of (processed_frame, detections, tracked_objects)
        """
        result = self._process_frame(frame)

        # Draw visualizations on the frame
        processed_frame = self._draw_frame_visualization(frame, result)

        return processed_frame, result["detections"], result["tracked_objects"]

    def get_traffic_statistics(self) -> Dict:
        """
        Get current traffic statistics

        Returns:
            Dictionary containing traffic statistics
        """
        return self.analytics.get_traffic_statistics()


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis using CCTV Footage"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Configuration file path"
    )
    parser.add_argument("--video", help="Input video file path")
    parser.add_argument("--camera", type=int, help="Camera device ID for live feed")
    parser.add_argument("--output", help="Output video file path")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = TrafficFlowAnalyzer(args.config)

    try:
        if args.camera is not None:
            # Process live feed
            analyzer.process_live_feed(args.camera)
        elif args.video:
            # Process video file
            results = analyzer.process_video(args.video, args.output)
            print("Analysis completed. Results saved to output directory.")

            # Print summary
            traffic_stats = results["traffic_statistics"]
            print("\nSummary:")
            print(
                f"- Total vehicles detected: {traffic_stats.get('total_vehicles_detected', 0)}"
            )
            print(
                f"- Analysis duration: {traffic_stats.get('analysis_duration', 0):.1f} seconds"
            )
            print(
                f"- Average vehicles per frame: {traffic_stats.get('average_vehicles_per_frame', 0):.1f}"
            )
        else:
            print("Please specify either --video or --camera option")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()
