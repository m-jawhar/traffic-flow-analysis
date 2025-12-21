"""
Traffic Analytics Module
Comprehensive traffic flow analysis and congestion estimation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import cv2
from datetime import datetime, timedelta
import json


class TrafficAnalyzer:
    """Main traffic analysis engine"""

    def __init__(self, config: Dict):
        """
        Initialize traffic analyzer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.analytics_config = config.get("analytics", {})

        # Congestion thresholds
        self.congestion_thresholds = self.analytics_config.get(
            "congestion_thresholds", {"low": 10, "medium": 25, "high": 50}
        )

        # Data storage
        self.vehicle_counts = defaultdict(int)
        self.vehicle_speeds = deque(maxlen=100)
        self.congestion_history = deque(maxlen=1000)
        self.traffic_flow_data = []

        # Counting zones
        self.counting_zones = self.analytics_config.get("counting_zones", [])

        # Analytics state
        self.frame_count = 0
        self.start_time = datetime.now()

    def analyze_frame(
        self, tracked_objects: Dict[int, Dict], frame_shape: Tuple[int, int]
    ) -> Dict:
        """
        Analyze single frame for traffic metrics

        Args:
            tracked_objects: Dictionary of tracked vehicles
            frame_shape: Shape of the frame (height, width)

        Returns:
            Frame analysis results
        """
        self.frame_count += 1
        current_time = datetime.now()

        # Count vehicles by type
        frame_counts = defaultdict(int)
        frame_speeds = []

        for obj_id, obj_info in tracked_objects.items():
            vehicle_type = obj_info["detection"]["class"]
            frame_counts[vehicle_type] += 1

            # Calculate speed if trajectory is long enough
            trajectory = obj_info["trajectory"]
            if len(trajectory) >= 5:  # Need at least 5 points for stable speed
                speed = self._calculate_vehicle_speed(trajectory)
                frame_speeds.append(speed)

        # Update global counts
        for vehicle_type, count in frame_counts.items():
            self.vehicle_counts[vehicle_type] += count

        # Update speeds
        if frame_speeds:
            self.vehicle_speeds.extend(frame_speeds)

        # Calculate congestion level
        total_vehicles = sum(frame_counts.values())
        congestion_level = self._calculate_congestion_level(total_vehicles, frame_shape)

        # Store congestion history
        self.congestion_history.append(
            {
                "timestamp": current_time,
                "level": congestion_level,
                "vehicle_count": total_vehicles,
            }
        )

        # Analyze traffic flow in zones
        zone_analysis = self._analyze_counting_zones(tracked_objects, frame_shape)

        # Calculate traffic density
        traffic_density = self._calculate_traffic_density(total_vehicles, frame_shape)

        analysis_result = {
            "frame_number": self.frame_count,
            "timestamp": current_time,
            "vehicle_counts": dict(frame_counts),
            "total_vehicles": total_vehicles,
            "congestion_level": congestion_level,
            "traffic_density": traffic_density,
            "average_speed": np.mean(frame_speeds) if frame_speeds else 0,
            "zone_analysis": zone_analysis,
            "cumulative_counts": dict(self.vehicle_counts),
        }

        # Store for historical analysis
        self.traffic_flow_data.append(analysis_result)

        return analysis_result

    def _calculate_vehicle_speed(self, trajectory: List[Tuple[int, int]]) -> float:
        """Calculate vehicle speed from trajectory"""
        if len(trajectory) < 2:
            return 0

        # Calculate distance traveled over last few points
        distances = []
        for i in range(1, min(5, len(trajectory))):
            p1 = trajectory[-(i + 1)]
            p2 = trajectory[-i]
            dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            distances.append(dist)

        return np.mean(distances) if distances else 0

    def _calculate_congestion_level(
        self, vehicle_count: int, frame_shape: Tuple[int, int]
    ) -> str:
        """Calculate congestion level based on vehicle density"""
        frame_area = frame_shape[0] * frame_shape[1]
        density = (vehicle_count * 10000) / frame_area  # Normalize by area

        if density <= self.congestion_thresholds["low"]:
            return "low"
        elif density <= self.congestion_thresholds["medium"]:
            return "medium"
        else:
            return "high"

    def _calculate_traffic_density(
        self, vehicle_count: int, frame_shape: Tuple[int, int]
    ) -> float:
        """Calculate traffic density (vehicles per unit area)"""
        frame_area = frame_shape[0] * frame_shape[1]
        return vehicle_count / (frame_area / 10000)  # Vehicles per 10k pixels

    def _analyze_counting_zones(
        self, tracked_objects: Dict[int, Dict], frame_shape: Tuple[int, int]
    ) -> Dict:
        """Analyze traffic flow in specific counting zones"""
        zone_results = {}

        for zone_idx, zone in enumerate(self.counting_zones):
            zone_name = zone.get("name", f"Zone_{zone_idx}")
            zone_polygon = zone.get("polygon", [])

            if not zone_polygon:
                continue

            # Count vehicles in zone
            vehicles_in_zone = 0
            for obj_id, obj_info in tracked_objects.items():
                centroid = obj_info["centroid"]
                if self._point_in_polygon(centroid, zone_polygon):
                    vehicles_in_zone += 1

            zone_results[zone_name] = {
                "vehicle_count": vehicles_in_zone,
                "density": vehicles_in_zone / len(zone_polygon) if zone_polygon else 0,
            }

        return zone_results

    def _point_in_polygon(
        self, point: Tuple[int, int], polygon: List[Tuple[int, int]]
    ) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_traffic_statistics(self) -> Dict:
        """Get comprehensive traffic statistics"""
        if not self.traffic_flow_data:
            return {}

        # Calculate statistics from historical data
        df = pd.DataFrame(self.traffic_flow_data)

        stats = {
            "analysis_duration": (datetime.now() - self.start_time).total_seconds(),
            "total_frames": self.frame_count,
            "total_vehicles_detected": sum(self.vehicle_counts.values()),
            "vehicle_breakdown": dict(self.vehicle_counts),
            "average_vehicles_per_frame": df["total_vehicles"].mean(),
            "peak_vehicle_count": df["total_vehicles"].max(),
            "average_speed": (
                np.mean(list(self.vehicle_speeds)) if self.vehicle_speeds else 0
            ),
            "congestion_distribution": {
                "low": len([x for x in self.congestion_history if x["level"] == "low"]),
                "medium": len(
                    [x for x in self.congestion_history if x["level"] == "medium"]
                ),
                "high": len(
                    [x for x in self.congestion_history if x["level"] == "high"]
                ),
            },
            "traffic_flow_trend": self._calculate_flow_trend(df),
            "busiest_periods": self._identify_busiest_periods(df),
        }

        return stats

    def _calculate_flow_trend(self, df: pd.DataFrame) -> str:
        """Calculate overall traffic flow trend"""
        if len(df) < 10:
            return "insufficient_data"

        # Simple trend analysis using linear regression
        x = np.arange(len(df))
        y = df["total_vehicles"].values

        slope = np.polyfit(x, y, 1)[0]

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _identify_busiest_periods(self, df: pd.DataFrame) -> List[Dict]:
        """Identify periods with highest traffic"""
        if len(df) < 20:
            return []

        # Find peaks in vehicle count
        vehicle_counts = df["total_vehicles"].values
        threshold = np.mean(vehicle_counts) + np.std(vehicle_counts)

        busy_periods = []
        in_busy_period = False
        period_start = None

        for i, count in enumerate(vehicle_counts):
            if count > threshold and not in_busy_period:
                in_busy_period = True
                period_start = i
            elif count <= threshold and in_busy_period:
                in_busy_period = False
                busy_periods.append(
                    {
                        "start_frame": period_start,
                        "end_frame": i,
                        "duration_frames": i - period_start,
                        "avg_vehicle_count": np.mean(vehicle_counts[period_start:i]),
                    }
                )

        return busy_periods[:5]  # Return top 5 busiest periods

    def export_analytics(self, filepath: str):
        """Export analytics data to file"""
        analytics_data = {
            "statistics": self.get_traffic_statistics(),
            "frame_data": self.traffic_flow_data,
            "config": self.config,
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(analytics_data, f, indent=2, default=str)

    def draw_analytics_overlay(
        self, frame: np.ndarray, analysis_result: Dict
    ) -> np.ndarray:
        """Draw analytics information on frame"""
        overlay_frame = frame.copy()

        # Draw text overlay with analytics info
        info_text = [
            f"Frame: {analysis_result['frame_number']}",
            f"Total Vehicles: {analysis_result['total_vehicles']}",
            f"Congestion: {analysis_result['congestion_level'].upper()}",
            f"Density: {analysis_result['traffic_density']:.2f}",
            f"Avg Speed: {analysis_result['average_speed']:.1f}",
        ]

        # Vehicle count breakdown
        for vehicle_type, count in analysis_result["vehicle_counts"].items():
            info_text.append(f"{vehicle_type.capitalize()}: {count}")

        # Draw background rectangle
        bg_height = len(info_text) * 25 + 20
        cv2.rectangle(overlay_frame, (10, 10), (300, bg_height), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (10, 10), (300, bg_height), (255, 255, 255), 2)

        # Draw text
        for i, text in enumerate(info_text):
            y_pos = 30 + i * 25
            cv2.putText(
                overlay_frame,
                text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        # Draw congestion level indicator
        congestion_color = {
            "low": (0, 255, 0),  # Green
            "medium": (0, 255, 255),  # Yellow
            "high": (0, 0, 255),  # Red
        }

        level = analysis_result["congestion_level"]
        color = congestion_color.get(level, (255, 255, 255))
        cv2.circle(overlay_frame, (frame.shape[1] - 30, 30), 20, color, -1)

        # Draw counting zones if configured
        for zone in self.counting_zones:
            polygon = zone.get("polygon", [])
            if polygon:
                pts = np.array(polygon, np.int32)
                cv2.polylines(overlay_frame, [pts], True, (255, 0, 255), 2)

        return overlay_frame
