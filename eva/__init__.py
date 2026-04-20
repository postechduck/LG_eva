"""EVA - Evaluation package for vehicle detection and tracking.

A modular package for evaluating vehicle detection and tracking
on the UA-DETRAC benchmark dataset.

Example usage:
    from eva import Config, EvaluationPipeline

    config = Config.from_yaml("configs/default.yaml")
    pipeline = EvaluationPipeline(config)
    results = pipeline.run_evaluation(max_sequences=5)
"""

# Version
__version__ = "1.0.0"

# Configuration
from .config import (
    Config,
    DatasetConfig,
    DetectionConfig,
    TrackingConfig,
    EvaluationConfig,
    OutputConfig,
    load_config,
)

# Data
from .data import (
    parse_ua_detrac_xml,
    UADETRACDataset,
    SequenceData,
)

# Models
from .models import (
    Detection,
    BaseDetector,
    RTDETRDetector,
    YOLODetector,
    create_detector,
    VEHICLE_CLASSES,
    TrackerArgs,
    Track,
    ByteTracker,
)

# Evaluation
from .evaluation import (
    DetectionMetrics,
    compute_ap,
    evaluate_detection_with_map,
    evaluate_detection_simple,
    TrackingMetrics,
    evaluate_tracking_single_sequence,
    evaluate_tracking_multi_sequence,
)

# Utilities
from .utils import (
    compute_iou,
    is_in_ignored_region,
    filter_ignored_boxes,
)

# Pipeline
from .pipeline import (
    FrameResult,
    SequenceResult,
    DetectionTrackingPipeline,
    EvaluationPipeline,
)

# Speed
from .speed import (
    SpeedClass,
    SpeedResult,
    TrackSpeedStats,
    calculate_pixel_speed,
    classify_speeds,
    compute_track_statistics,
    analyze_speed_distribution,
    get_suggested_thresholds,
    THRESHOLD_UNIFORM,
    THRESHOLD_SEMANTIC,
)

# Speed Evaluation
from .speed_evaluation import (
    SpeedComparisonResult,
    SpeedEvaluationMetrics,
    calculate_gt_speeds,
    evaluate_speed_accuracy,
    print_confusion_matrix,
    print_speed_evaluation_summary,
)

__all__ = [
    # Version
    '__version__',
    # Config
    'Config',
    'DatasetConfig',
    'DetectionConfig',
    'TrackingConfig',
    'EvaluationConfig',
    'OutputConfig',
    'load_config',
    # Data
    'parse_ua_detrac_xml',
    'UADETRACDataset',
    'SequenceData',
    # Models - Detection
    'Detection',
    'BaseDetector',
    'RTDETRDetector',
    'YOLODetector',
    'create_detector',
    'VEHICLE_CLASSES',
    # Models - Tracking
    'TrackerArgs',
    'Track',
    'ByteTracker',
    # Evaluation - Detection
    'DetectionMetrics',
    'compute_ap',
    'evaluate_detection_with_map',
    'evaluate_detection_simple',
    # Evaluation - Tracking
    'TrackingMetrics',
    'evaluate_tracking_single_sequence',
    'evaluate_tracking_multi_sequence',
    # Utilities
    'compute_iou',
    'is_in_ignored_region',
    'filter_ignored_boxes',
    # Pipeline
    'FrameResult',
    'SequenceResult',
    'DetectionTrackingPipeline',
    'EvaluationPipeline',
    # Speed
    'SpeedClass',
    'SpeedResult',
    'TrackSpeedStats',
    'calculate_pixel_speed',
    'classify_speeds',
    'compute_track_statistics',
    'analyze_speed_distribution',
    'get_suggested_thresholds',
]
