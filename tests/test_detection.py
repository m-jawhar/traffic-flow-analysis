"""
Tests for vehicle detection module
"""

import pytest
import cv2
import numpy as np
from src.detection.vehicle_detector import VehicleDetector


@pytest.fixture
def detector():
    """Create detector instance"""
    return VehicleDetector("config/config.yaml")


@pytest.fixture
def sample_frame():
    """Create sample frame for testing"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = [50, 50, 50]
    return frame


def test_detector_initialization(detector):
    """Test detector initialization"""
    assert detector is not None
    assert detector.model is not None
    assert detector.confidence_threshold > 0


def test_detect_vehicles(detector, sample_frame):
    """Test vehicle detection"""
    detections = detector.detect_vehicles(sample_frame)
    assert isinstance(detections, list)


def test_detection_format(detector, sample_frame):
    """Test detection output format"""
    detections = detector.detect_vehicles(sample_frame)

    for detection in detections:
        assert "bbox" in detection
        assert "confidence" in detection
        assert "class" in detection
        assert len(detection["bbox"]) == 4


def test_draw_detections(detector, sample_frame):
    """Test drawing detections on frame"""
    detections = [
        {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.85,
            "class": "car",
            "class_id": 2,
        }
    ]

    vis_frame = detector.draw_detections(sample_frame, detections)
    assert vis_frame.shape == sample_frame.shape
    assert isinstance(vis_frame, np.ndarray)
    # Check that the function returns a frame (basic functionality test)
    assert vis_frame is not None


def test_get_detection_centers(detector):
    """Test getting detection centers"""
    detections = [
        {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.85,
            "class": "car",
            "class_id": 2,
        },
        {
            "bbox": [300, 150, 400, 250],
            "confidence": 0.90,
            "class": "bus",
            "class_id": 5,
        },
    ]

    centers = detector.get_detection_centers(detections)
    assert len(centers) == 2
    assert centers[0] == (150, 150)  # Center of first bbox
    assert centers[1] == (350, 200)  # Center of second bbox
