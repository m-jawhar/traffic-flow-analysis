"""
Vehicle Classification Module
Advanced classification and feature extraction for detected vehicles
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from typing import List, Dict, Tuple
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class VehicleClassifier:
    """Enhanced vehicle classification with feature extraction"""

    def __init__(self):
        """Initialize the vehicle classifier"""
        self.feature_extractor = FeatureExtractor()
        self.ml_classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features(self, image_crop: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Extract features from vehicle crop

        Args:
            image_crop: Cropped vehicle image
            detection: Detection dictionary

        Returns:
            Feature vector
        """
        features = []

        # Basic geometric features
        bbox = detection["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 0
        area = width * height

        features.extend([width, height, aspect_ratio, area])

        # Color features
        color_features = self._extract_color_features(image_crop)
        features.extend(color_features)

        # Texture features
        texture_features = self._extract_texture_features(image_crop)
        features.extend(texture_features)

        # Shape features
        shape_features = self._extract_shape_features(image_crop)
        features.extend(shape_features)

        return np.array(features)

    def _extract_color_features(self, image: np.ndarray) -> List[float]:
        """Extract color-based features"""
        if len(image.shape) == 3:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # Calculate mean and std for each channel
            features = []
            for img, name in [(image, "bgr"), (hsv, "hsv"), (lab, "lab")]:
                for channel in range(3):
                    features.append(np.mean(img[:, :, channel]))
                    features.append(np.std(img[:, :, channel]))
        else:
            # Grayscale image
            features = [np.mean(image), np.std(image)] * 9  # Pad to match color

        return features

    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture features using Local Binary Patterns"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Simple texture measures
        features = []

        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))

        # Laplacian (edge detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(np.mean(np.abs(laplacian)))
        features.append(np.std(laplacian))

        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        features.extend(hist.flatten()[:10])  # Use first 10 bins

        return features

    def _extract_shape_features(self, image: np.ndarray) -> List[float]:
        """Extract shape-based features"""
        # Convert to grayscale and binary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        features = []
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Contour area
            area = cv2.contourArea(largest_contour)
            features.append(area)

            # Perimeter
            perimeter = cv2.arcLength(largest_contour, True)
            features.append(perimeter)

            # Compactness
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter**2)
            else:
                compactness = 0
            features.append(compactness)

            # Bounding rectangle features
            x, y, w, h = cv2.boundingRect(largest_contour)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            features.append(extent)

        else:
            features = [0, 0, 0, 0]

        return features

    def classify_vehicle(self, image_crop: np.ndarray, detection: Dict) -> Dict:
        """
        Classify vehicle type with confidence

        Args:
            image_crop: Cropped vehicle image
            detection: Detection dictionary

        Returns:
            Classification result with confidence
        """
        # Extract features
        features = self.extract_features(image_crop, detection)

        if self.is_trained and self.ml_classifier:
            # Use ML classifier if trained
            features_scaled = self.scaler.transform([features])
            prediction = self.ml_classifier.predict(features_scaled)[0]
            probabilities = self.ml_classifier.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
        else:
            # Use rule-based classification
            prediction, confidence = self._rule_based_classification(
                features, detection
            )

        return {
            "predicted_class": prediction,
            "confidence": confidence,
            "features": features,
        }

    def _rule_based_classification(
        self, features: np.ndarray, detection: Dict
    ) -> Tuple[str, float]:
        """Rule-based classification fallback"""
        bbox = detection["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 0
        area = width * height

        # Simple rules based on size and aspect ratio
        if aspect_ratio > 2.5:  # Very wide
            return "bus", 0.7
        elif aspect_ratio < 0.8:  # Tall
            return "motorcycle", 0.6
        elif area > 15000:  # Large
            if aspect_ratio > 1.8:
                return "bus", 0.8
            else:
                return "truck", 0.7
        elif area < 3000:  # Small
            return "bicycle", 0.6
        else:  # Medium size
            return "car", 0.8

    def train_classifier(self, training_data: List[Dict]):
        """
        Train ML classifier on labeled data

        Args:
            training_data: List of training samples with features and labels
        """
        if not training_data:
            return

        # Prepare training data
        X = []
        y = []

        for sample in training_data:
            X.append(sample["features"])
            y.append(sample["label"])

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest classifier
        self.ml_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.ml_classifier.fit(X_scaled, y)
        self.is_trained = True

    def save_model(self, filepath: str):
        """Save trained model"""
        if self.is_trained:
            model_data = {
                "classifier": self.ml_classifier,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
            }
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self.ml_classifier = model_data["classifier"]
            self.scaler = model_data["scaler"]
            self.is_trained = model_data["is_trained"]
        except FileNotFoundError:
            print(f"Model file {filepath} not found. Using rule-based classification.")


class FeatureExtractor:
    """Deep learning based feature extractor"""

    def __init__(self):
        """Initialize feature extractor"""
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep learning features (placeholder for future implementation)"""
        # This would use a pre-trained CNN for feature extraction
        # For now, return basic features
        return np.random.rand(512)  # Placeholder
