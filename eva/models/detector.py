"""Detection model wrappers for EVA.

Provides unified interface for RT-DETR and YOLO detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# COCO vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


@dataclass
class Detection:
    """Single detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def box(self) -> Tuple[float, float, float, float]:
        """Get box as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def tlbr(self) -> np.ndarray:
        """Get box as numpy array [x1, y1, x2, y2]."""
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def to_bytetrack_format(self) -> List[float]:
        """Convert to ByteTrack input format [x1, y1, x2, y2, conf]."""
        return [self.x1, self.y1, self.x2, self.y2, self.confidence]


class BaseDetector(ABC):
    """Abstract base class for detectors."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.3,
        vehicle_classes: Optional[List[int]] = None,
        verbose: bool = False
    ):
        """Initialize detector.

        Args:
            model_path: Path to model weights.
            confidence_threshold: Minimum confidence threshold.
            vehicle_classes: List of class IDs to detect (default: vehicles).
            verbose: Whether to print verbose output.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.vehicle_classes = vehicle_classes or VEHICLE_CLASSES
        self.verbose = verbose
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a frame.

        Args:
            frame: BGR image frame.

        Returns:
            List of Detection objects.
        """
        pass

    def warmup(self, image_size: Tuple[int, int] = (540, 960)) -> None:
        """Warm up the model with a dummy inference.

        Args:
            image_size: (height, width) for dummy image.
        """
        dummy = np.zeros((*image_size, 3), dtype=np.uint8)
        self.detect(dummy)


class RTDETRDetector(BaseDetector):
    """RT-DETR detector wrapper."""

    def load_model(self) -> None:
        """Load RT-DETR model."""
        from ultralytics import RTDETR
        self.model = RTDETR(self.model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run RT-DETR detection.

        Args:
            frame: BGR image frame.

        Returns:
            List of Detection objects.
        """
        if self.model is None:
            self.load_model()

        results = self.model(
            frame,
            verbose=self.verbose,
            conf=self.confidence_threshold
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append(Detection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=conf,
                        class_id=cls
                    ))

        return detections


class YOLODetector(BaseDetector):
    """YOLO detector wrapper."""

    def load_model(self) -> None:
        """Load YOLO model."""
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO detection.

        Args:
            frame: BGR image frame.

        Returns:
            List of Detection objects.
        """
        if self.model is None:
            self.load_model()

        results = self.model(
            frame,
            verbose=self.verbose,
            conf=self.confidence_threshold
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append(Detection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=conf,
                        class_id=cls
                    ))

        return detections


def create_detector(
    model_type: str,
    model_path: str,
    confidence_threshold: float = 0.3,
    vehicle_classes: Optional[List[int]] = None,
    verbose: bool = False
) -> BaseDetector:
    """Factory function to create a detector.

    Args:
        model_type: Type of detector ("rtdetr" or "yolo").
        model_path: Path to model weights.
        confidence_threshold: Minimum confidence threshold.
        vehicle_classes: List of class IDs to detect.
        verbose: Whether to print verbose output.

    Returns:
        Detector instance.

    Raises:
        ValueError: If model_type is not supported.
    """
    model_type = model_type.lower()

    if model_type == "rtdetr":
        detector = RTDETRDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            vehicle_classes=vehicle_classes,
            verbose=verbose
        )
    elif model_type in ("yolo", "yolov8"):
        detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            vehicle_classes=vehicle_classes,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    detector.load_model()
    return detector
