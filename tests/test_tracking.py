"""
Tests for vehicle tracking module
"""

import pytest
import numpy as np
from src.tracking.vehicle_tracker import VehicleTracker, CentroidTracker


@pytest.fixture
def tracker():
    """Create tracker instance"""
    return VehicleTracker(tracker_type="centroid", max_disappeared=30, max_distance=100)


@pytest.fixture
def sample_detections():
    """Create sample detections"""
    return [
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


def test_tracker_initialization(tracker):
    """Test tracker initialization"""
    assert tracker is not None
    assert tracker.tracker is not None


def test_tracker_update(tracker, sample_detections):
    """Test tracker update"""
    tracked_objects = tracker.update(sample_detections)
    assert isinstance(tracked_objects, dict)
    assert len(tracked_objects) == 2


def test_object_registration(tracker, sample_detections):
    """Test object registration"""
    tracked_objects = tracker.update(sample_detections)

    # Check object IDs
    assert 0 in tracked_objects
    assert 1 in tracked_objects

    # Check object information
    for obj_id, obj_info in tracked_objects.items():
        assert "centroid" in obj_info
        assert "detection" in obj_info
        assert "trajectory" in obj_info


def test_trajectory_tracking(tracker, sample_detections):
    """Test trajectory tracking across frames"""
    # First frame
    tracked_objects_1 = tracker.update(sample_detections)

    # Second frame with moved detections
    moved_detections = [
        {
            "bbox": [110, 100, 210, 200],
            "confidence": 0.85,
            "class": "car",
            "class_id": 2,
        },
        {
            "bbox": [310, 150, 410, 250],
            "confidence": 0.90,
            "class": "bus",
            "class_id": 5,
        },
    ]

    tracked_objects_2 = tracker.update(moved_detections)

    # Check trajectory length increased
    for obj_id, obj_info in tracked_objects_2.items():
        assert len(obj_info["trajectory"]) >= 2


def test_object_disappearance(tracker, sample_detections):
    """Test object disappearance handling"""
    # Register objects
    tracked_objects_1 = tracker.update(sample_detections)
    initial_count = len(tracked_objects_1)

    # Update with empty detections multiple times
    for _ in range(35):  # More than max_disappeared
        tracked_objects = tracker.update([])

    # Objects should be deregistered
    assert len(tracked_objects) < initial_count
