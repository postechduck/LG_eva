"""Configuration management for EVA.

Provides Config dataclass and YAML loading functionality.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    base_path: str = "/data/home/pkw_aim25/EVA_2/data/UA-DETRAC"
    image_dir: str = "DETRAC-Images"
    train_annotation_dir: str = "DETRAC-Train-Annotations-XML"
    test_annotation_dir: str = "DETRAC-Test-Annotations-XML"
    image_extension: str = ".jpg"
    frame_width: int = 960
    frame_height: int = 540
    fps: float = 25.0


@dataclass
class DetectionConfig:
    """Detection model configuration."""
    model_type: str = "rtdetr"
    model_path: str = "/data/home/pkw_aim25/EVA_2/models/detection/rtdetr-l.pt"
    confidence_threshold: float = 0.3
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    verbose: bool = False


@dataclass
class TrackingConfig:
    """Tracking model configuration."""
    # Common
    tracker_type: str = "bytetrack"  # "bytetrack" or "deepsort"

    # ByteTrack parameters
    track_thresh: float = 0.3
    track_buffer: int = 30
    match_thresh: float = 0.8
    mot20: bool = False

    # DeepSORT parameters
    reid_model_path: str = ""  # Re-ID 모델 경로
    max_dist: float = 0.2      # 최대 cosine 거리
    max_iou_distance: float = 0.7
    max_age: int = 70          # 최대 lost 프레임 수
    n_init: int = 3            # 확정까지 필요한 연속 매칭 수
    nn_budget: int = 100       # feature 저장 개수


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75])
    iou_threshold_range_start: float = 0.5
    iou_threshold_range_end: float = 0.95
    iou_threshold_range_step: float = 0.05
    use_ignored_regions: bool = True
    tracking_iou_threshold: float = 0.5


@dataclass
class OutputConfig:
    """Output configuration."""
    base_path: str = "/data/home/pkw_aim25/EVA_2/outputs"
    save_results: bool = True
    save_videos: bool = False
    video_fps: int = 25


@dataclass
class Config:
    """Main configuration container."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Config instance.
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            Config instance.
        """
        config = cls()

        if 'dataset' in data:
            config.dataset = DatasetConfig(**data['dataset'])

        if 'detection' in data:
            config.detection = DetectionConfig(**data['detection'])

        if 'tracking' in data:
            config.tracking = TrackingConfig(**data['tracking'])

        if 'evaluation' in data:
            eval_data = data['evaluation'].copy()
            # Handle nested iou_threshold_range
            if 'iou_threshold_range' in eval_data:
                range_data = eval_data.pop('iou_threshold_range')
                eval_data['iou_threshold_range_start'] = range_data.get('start', 0.5)
                eval_data['iou_threshold_range_end'] = range_data.get('end', 0.95)
                eval_data['iou_threshold_range_step'] = range_data.get('step', 0.05)
            config.evaluation = EvaluationConfig(**eval_data)

        if 'output' in data:
            config.output = OutputConfig(**data['output'])

        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            'dataset': {
                'base_path': self.dataset.base_path,
                'image_dir': self.dataset.image_dir,
                'train_annotation_dir': self.dataset.train_annotation_dir,
                'test_annotation_dir': self.dataset.test_annotation_dir,
                'image_extension': self.dataset.image_extension,
                'frame_width': self.dataset.frame_width,
                'frame_height': self.dataset.frame_height,
                'fps': self.dataset.fps,
            },
            'detection': {
                'model_type': self.detection.model_type,
                'model_path': self.detection.model_path,
                'confidence_threshold': self.detection.confidence_threshold,
                'vehicle_classes': self.detection.vehicle_classes,
                'verbose': self.detection.verbose,
            },
            'tracking': {
                'tracker_type': self.tracking.tracker_type,
                # ByteTrack
                'track_thresh': self.tracking.track_thresh,
                'track_buffer': self.tracking.track_buffer,
                'match_thresh': self.tracking.match_thresh,
                'mot20': self.tracking.mot20,
                # DeepSORT
                'reid_model_path': self.tracking.reid_model_path,
                'max_dist': self.tracking.max_dist,
                'max_iou_distance': self.tracking.max_iou_distance,
                'max_age': self.tracking.max_age,
                'n_init': self.tracking.n_init,
                'nn_budget': self.tracking.nn_budget,
            },
            'evaluation': {
                'iou_thresholds': self.evaluation.iou_thresholds,
                'iou_threshold_range': {
                    'start': self.evaluation.iou_threshold_range_start,
                    'end': self.evaluation.iou_threshold_range_end,
                    'step': self.evaluation.iou_threshold_range_step,
                },
                'use_ignored_regions': self.evaluation.use_ignored_regions,
                'tracking_iou_threshold': self.evaluation.tracking_iou_threshold,
            },
            'output': {
                'base_path': self.output.base_path,
                'save_results': self.output.save_results,
                'save_videos': self.output.save_videos,
                'video_fps': self.output.video_fps,
            },
        }

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file.
        """
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Config instance.
    """
    if config_path is None:
        return Config()

    path = Path(config_path)
    if not path.exists():
        print(f"Warning: Config file not found: {config_path}. Using defaults.")
        return Config()

    return Config.from_yaml(str(path))
