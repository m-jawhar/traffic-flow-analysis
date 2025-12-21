"""
Vehicle Detection Module using YOLO
Handles real-time vehicle detection in video frames
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict
import yaml


class VehicleDetector:
    """YOLO-based vehicle detector"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the vehicle detector

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.detection_config = self.config["detection"]
        self.classification_config = self.config["classification"]

        # Load YOLO model
        self.model = YOLO(self.detection_config["model_path"])

        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = {
            "car": 2,
            "motorcycle": 3,
            "bus": 5,
            "truck": 7,
            "bicycle": 1,
        }

        self.confidence_threshold = self.detection_config["confidence_threshold"]
        self.nms_threshold = self.detection_config["nms_threshold"]

    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a single frame

        Args:
            frame: Input frame as numpy array

        Returns:
            List of detection dictionaries containing bbox, class, confidence
        """
        # Run inference
        results = self.model(
            frame, conf=self.confidence_threshold, iou=self.nms_threshold
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # Check if it's a vehicle class
                    class_name = self.model.names[class_id]
                    if (
                        class_name in self.vehicle_classes.values()
                        or class_id in self.vehicle_classes.values()
                    ):
                        # Map to our vehicle categories
                        vehicle_type = self._map_to_vehicle_type(class_id, class_name)

                        detection = {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(confidence),
                            "class": vehicle_type,
                            "class_id": class_id,
                        }
                        detections.append(detection)

        return detections

    def _map_to_vehicle_type(self, class_id: int, class_name: str) -> str:
        """Map COCO class to vehicle type"""
        mapping = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
        return mapping.get(class_id, class_name)

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame

        Args:
            frame: Input frame
            detections: List of detection dictionaries

        Returns:
            Frame with drawn detections
        """
        colors = self.config["visualization"]["colors"]

        for detection in detections:
            bbox = detection["bbox"]
            vehicle_type = detection["class"]
            confidence = detection["confidence"]

            # Get color for vehicle type
            color = colors.get(vehicle_type, [255, 255, 255])

            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label
            label = f"{vehicle_type}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1] - label_size[1] - 10),
                (bbox[0] + label_size[0], bbox[1]),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return frame

    def get_detection_centers(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """Get center points of all detections"""
        centers = []
        for detection in detections:
            bbox = detection["bbox"]
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            centers.append((center_x, center_y))
        return centers
