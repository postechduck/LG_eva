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
    DeepSORTArgs,
    Track,
    ByteTracker,
    DeepSORTTracker,
    create_tracker,
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
    'DeepSORTArgs',
    'Track',
    'ByteTracker',
    'DeepSORTTracker',
    'create_tracker',
]
