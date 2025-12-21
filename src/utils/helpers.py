"""
Utility functions for traffic flow analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import yaml
import os
from datetime import datetime


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_output_directories(base_dir: str) -> Dict[str, str]:
    """Create output directories for saving results"""
    dirs = {
        "videos": os.path.join(base_dir, "videos"),
        "analytics": os.path.join(base_dir, "analytics"),
        "images": os.path.join(base_dir, "images"),
        "logs": os.path.join(base_dir, "logs"),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def resize_frame(frame: np.ndarray, target_width: int = 1280) -> np.ndarray:
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame

    scale = target_width / width
    new_width = target_width
    new_height = int(height * scale)

    return cv2.resize(frame, (new_width, new_height))


def draw_counting_zone(
    frame: np.ndarray,
    zone_points: List[Tuple[int, int]],
    zone_name: str = "Zone",
    color: Tuple[int, int, int] = (255, 0, 255),
) -> np.ndarray:
    """Draw counting zone on frame"""
    if not zone_points:
        return frame

    # Draw zone polygon
    pts = np.array(zone_points, np.int32)
    cv2.polylines(frame, [pts], True, color, 2)

    # Fill zone with transparent overlay
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

    # Draw zone name
    centroid_x = int(np.mean([pt[0] for pt in zone_points]))
    centroid_y = int(np.mean([pt[1] for pt in zone_points]))
    cv2.putText(
        frame,
        zone_name,
        (centroid_x - 30, centroid_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return frame


def calculate_fps(start_time: float, frame_count: int) -> float:
    """Calculate frames per second"""
    elapsed_time = cv2.getTickCount() - start_time
    elapsed_time /= cv2.getTickFrequency()
    return frame_count / elapsed_time if elapsed_time > 0 else 0


def create_video_writer(
    output_path: str, fps: int, frame_size: Tuple[int, int], fourcc: str = "mp4v"
) -> cv2.VideoWriter:
    """Create video writer for saving output"""
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    return cv2.VideoWriter(output_path, fourcc_code, fps, frame_size)


def log_message(message: str, log_file: Optional[str] = None):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"

    print(log_entry)

    if log_file:
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")


def validate_video_source(source) -> bool:
    """Validate video source (file or camera)"""
    if isinstance(source, str):
        return os.path.exists(source)
    elif isinstance(source, int):
        cap = cv2.VideoCapture(source)
        is_valid = cap.isOpened()
        cap.release()
        return is_valid
    return False


def get_video_properties(video_path: str) -> Dict:
    """Get video properties"""
    cap = cv2.VideoCapture(video_path)

    properties = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
    }

    cap.release()
    return properties


def crop_vehicle_from_frame(
    frame: np.ndarray, bbox: List[int], padding: int = 10
) -> np.ndarray:
    """Crop vehicle region from frame with padding"""
    x1, y1, x2, y2 = bbox

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)

    return frame[y1:y2, x1:x2]


def calculate_intersection_over_union(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def smooth_trajectory(
    trajectory: List[Tuple[int, int]], window_size: int = 5
) -> List[Tuple[int, int]]:
    """Smooth trajectory using moving average"""
    if len(trajectory) < window_size:
        return trajectory

    smoothed = []
    for i in range(len(trajectory)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(trajectory), i + window_size // 2 + 1)

        window_points = trajectory[start_idx:end_idx]
        avg_x = sum(pt[0] for pt in window_points) / len(window_points)
        avg_y = sum(pt[1] for pt in window_points) / len(window_points)

        smoothed.append((int(avg_x), int(avg_y)))

    return smoothed


def convert_speed_to_kmh(
    pixel_speed: float, meters_per_pixel: float, fps: float
) -> float:
    """Convert pixel speed to km/h"""
    # pixel_speed is pixels per frame
    # Convert to pixels per second, then to meters per second, then to km/h
    pixels_per_second = pixel_speed * fps
    meters_per_second = pixels_per_second * meters_per_pixel
    kmh = meters_per_second * 3.6
    return kmh


def create_heatmap(
    points: List[Tuple[int, int]], frame_shape: Tuple[int, int], blur_size: int = 15
) -> np.ndarray:
    """Create heatmap from list of points"""
    height, width = frame_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1

    # Apply Gaussian blur
    heatmap = cv2.GaussianBlur(heatmap, (blur_size, blur_size), 0)

    # Normalize to 0-255
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

    return heatmap


def apply_colormap_to_heatmap(
    heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Apply colormap to heatmap"""
    return cv2.applyColorMap(heatmap, colormap)


def overlay_heatmap_on_frame(
    frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Overlay heatmap on frame"""
    colored_heatmap = apply_colormap_to_heatmap(heatmap)
    return cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)


class PerformanceMonitor:
    """Monitor performance metrics"""

    def __init__(self):
        self.start_time = cv2.getTickCount()
        self.frame_count = 0
        self.processing_times = []

    def update(self, processing_time: float):
        """Update performance metrics"""
        self.frame_count += 1
        self.processing_times.append(processing_time)

    def get_fps(self) -> float:
        """Get current FPS"""
        elapsed = (cv2.getTickCount() - self.start_time) / cv2.getTickFrequency()
        return self.frame_count / elapsed if elapsed > 0 else 0

    def get_average_processing_time(self) -> float:
        """Get average processing time per frame"""
        return np.mean(self.processing_times) if self.processing_times else 0

    def reset(self):
        """Reset performance counters"""
        self.start_time = cv2.getTickCount()
        self.frame_count = 0
        self.processing_times = []
