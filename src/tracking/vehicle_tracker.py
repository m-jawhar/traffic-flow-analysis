"""
Vehicle Tracking Module
Implements multiple tracking algorithms for vehicle motion tracking
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import math


class CentroidTracker:
    """Simple centroid-based object tracker"""

    def __init__(self, max_disappeared: int = 50, max_distance: int = 100):
        """
        Initialize the centroid tracker

        Args:
            max_disappeared: Maximum frames an object can disappear before removal
            max_distance: Maximum distance for object association
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int], detection: Dict) -> int:
        """Register a new object"""
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "detection": detection,
            "trajectory": [centroid],
            "frame_count": 1,
        }
        self.disappeared[self.next_object_id] = 0
        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id: int):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with new detections

        Args:
            detections: List of detection dictionaries

        Returns:
            Dictionary mapping object IDs to object info
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Get centroids from detections
        input_centroids = []
        for detection in detections:
            bbox = detection["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            # Register all new objects
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, detections[i])
        else:
            # Match existing objects with new detections
            object_centroids = [obj["centroid"] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())

            # Compute distance matrix
            distances = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
                axis=2,
            )

            # Find minimum distance associations
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for row, col in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if distances[row, col] > self.max_distance:
                    continue

                # Update existing object
                object_id = object_ids[row]
                self.objects[object_id]["centroid"] = input_centroids[col]
                self.objects[object_id]["detection"] = detections[col]
                self.objects[object_id]["trajectory"].append(input_centroids[col])
                self.objects[object_id]["frame_count"] += 1
                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            # Handle unmatched detections and objects
            unused_rows = set(range(0, distances.shape[0])).difference(used_row_indices)
            unused_cols = set(range(0, distances.shape[1])).difference(used_col_indices)

            if distances.shape[0] >= distances.shape[1]:
                # More objects than detections - mark objects as disappeared
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than objects - register new objects
                for col in unused_cols:
                    self.register(input_centroids[col], detections[col])

        return self.objects


class VehicleTracker:
    """Main vehicle tracking class"""

    def __init__(self, tracker_type: str = "centroid", **kwargs):
        """
        Initialize vehicle tracker

        Args:
            tracker_type: Type of tracker to use ("centroid" or "deepsort")
            **kwargs: Additional arguments for tracker
        """
        self.tracker_type = tracker_type

        if tracker_type == "centroid":
            self.tracker = CentroidTracker(**kwargs)
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")

        self.vehicle_speeds = {}
        self.frame_count = 0

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """Update tracker with new detections"""
        self.frame_count += 1
        tracked_objects = self.tracker.update(detections)

        # Calculate speeds for tracked objects
        self._calculate_speeds(tracked_objects)

        return tracked_objects

    def _calculate_speeds(self, tracked_objects: Dict[int, Dict]):
        """Calculate vehicle speeds based on trajectory"""
        for object_id, obj_info in tracked_objects.items():
            trajectory = obj_info["trajectory"]

            if len(trajectory) >= 2:
                # Calculate speed based on last two positions
                p1 = trajectory[-2]
                p2 = trajectory[-1]

                # Pixel distance
                pixel_distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

                # Store speed (pixels per frame - can be converted to real units)
                self.vehicle_speeds[object_id] = pixel_distance

    def draw_tracks(
        self, frame: np.ndarray, tracked_objects: Dict[int, Dict]
    ) -> np.ndarray:
        """
        Draw tracking information on frame

        Args:
            frame: Input frame
            tracked_objects: Dictionary of tracked objects

        Returns:
            Frame with tracking visualization
        """
        for object_id, obj_info in tracked_objects.items():
            centroid = obj_info["centroid"]
            trajectory = obj_info["trajectory"]
            detection = obj_info["detection"]

            # Draw trajectory
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 2)

            # Draw centroid
            cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

            # Draw object ID
            cv2.putText(
                frame,
                f"ID: {object_id}",
                (centroid[0] - 20, centroid[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Draw speed if available
            if object_id in self.vehicle_speeds:
                speed = self.vehicle_speeds[object_id]
                cv2.putText(
                    frame,
                    f"Speed: {speed:.1f}",
                    (centroid[0] - 20, centroid[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

        return frame

    def get_track_statistics(self) -> Dict:
        """Get tracking statistics"""
        return {
            "total_objects_tracked": self.tracker.next_object_id,
            "current_objects": len(self.tracker.objects),
            "frame_count": self.frame_count,
            "average_speed": (
                np.mean(list(self.vehicle_speeds.values()))
                if self.vehicle_speeds
                else 0
            ),
        }
