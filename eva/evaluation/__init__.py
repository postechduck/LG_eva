"""EVA evaluation package."""

from .detection import (
    DetectionMetrics,
    compute_ap,
    evaluate_detection_with_map,
    evaluate_detection_simple,
)
from .tracking import (
    TrackingMetrics,
    evaluate_tracking_single_sequence,
    evaluate_tracking_multi_sequence,
)

__all__ = [
    # Detection
    'DetectionMetrics',
    'compute_ap',
    'evaluate_detection_with_map',
    'evaluate_detection_simple',
    # Tracking
    'TrackingMetrics',
    'evaluate_tracking_single_sequence',
    'evaluate_tracking_multi_sequence',
]
