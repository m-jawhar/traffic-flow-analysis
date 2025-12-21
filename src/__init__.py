"""
Traffic Flow Analysis System
A comprehensive computer vision solution for analyzing traffic patterns from CCTV footage
"""

__version__ = "1.0.0"
__author__ = "Traffic Analysis Team"

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.classification.vehicle_classifier import VehicleClassifier
from src.analytics.traffic_analyzer import TrafficAnalyzer
from src.visualization.traffic_visualizer import TrafficVisualizer

__all__ = [
    "VehicleDetector",
    "VehicleTracker",
    "VehicleClassifier",
    "TrafficAnalyzer",
    "TrafficVisualizer",
]
