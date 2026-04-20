"""EVA models package."""

from .detector import (
    Detection,
    BaseDetector,
    RTDETRDetector,
    YOLODetector,
    create_detector,
    VEHICLE_CLASSES,
)
from .tracker import (
    TrackerArgs,
    Track,
    ByteTracker,
)

__all__ = [
    # Detector
    'Detection',
    'BaseDetector',
    'RTDETRDetector',
    'YOLODetector',
    'create_detector',
    'VEHICLE_CLASSES',
    # Tracker
    'TrackerArgs',
    'Track',
    'ByteTracker',
]
