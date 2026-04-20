"""Detection and Tracking Pipeline for EVA.

Provides unified pipeline for running detection and tracking.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .data import UADETRACDataset, SequenceData
from .models import (
    Detection, Track,
    create_detector, BaseDetector,
    TrackerArgs, ByteTracker
)
from .evaluation import (
    DetectionMetrics, TrackingMetrics,
    evaluate_detection_with_map,
    evaluate_tracking_multi_sequence
)


@dataclass
class FrameResult:
    """Result for a single frame."""
    frame_num: int
    detections: List[Detection]
    tracks: List[Track]
    detection_latency_ms: float = 0.0
    tracking_latency_ms: float = 0.0

    @property
    def latency_ms(self) -> float:
        """Total latency (detection + tracking)."""
        return self.detection_latency_ms + self.tracking_latency_ms

    @property
    def total_latency_ms(self) -> float:
        """Total latency (detection + tracking)."""
        return self.detection_latency_ms + self.tracking_latency_ms

    def get_track_tuples(self, include_score: bool = True) -> List[Tuple]:
        """Get tracks as list of tuples.

        Args:
            include_score: Whether to include confidence score.

        Returns:
            List of (track_id, x1, y1, x2, y2[, score]) tuples.
        """
        return [t.to_tuple(include_score) for t in self.tracks]


@dataclass
class SequenceResult:
    """Result for a sequence."""
    seq_name: str
    predictions: Dict[int, List[Tuple]]
    num_frames: int
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0


class DetectionTrackingPipeline:
    """Combined detection and tracking pipeline."""

    def __init__(
        self,
        detector: BaseDetector,
        tracker_args: Optional[TrackerArgs] = None,
        warmup: bool = True
    ):
        """Initialize pipeline.

        Args:
            detector: Detector instance.
            tracker_args: Optional tracker configuration.
            warmup: Whether to warm up the detector.
        """
        self.detector = detector
        self.tracker_args = tracker_args or TrackerArgs()
        self.tracker = None

        if warmup:
            self.detector.warmup()

    @classmethod
    def from_config(cls, config: Config) -> 'DetectionTrackingPipeline':
        """Create pipeline from configuration.

        Args:
            config: Config instance.

        Returns:
            DetectionTrackingPipeline instance.
        """
        detector = create_detector(
            model_type=config.detection.model_type,
            model_path=config.detection.model_path,
            confidence_threshold=config.detection.confidence_threshold,
            vehicle_classes=config.detection.vehicle_classes,
            verbose=config.detection.verbose
        )

        tracker_args = TrackerArgs(
            track_thresh=config.tracking.track_thresh,
            track_buffer=config.tracking.track_buffer,
            match_thresh=config.tracking.match_thresh,
            mot20=config.tracking.mot20
        )

        return cls(detector=detector, tracker_args=tracker_args)

    def reset_tracker(self) -> None:
        """Reset the tracker for a new sequence."""
        self.tracker = ByteTracker(self.tracker_args)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int = 1
    ) -> FrameResult:
        """Process a single frame.

        Args:
            frame: BGR image frame.
            frame_num: Frame number.

        Returns:
            FrameResult with detections and tracks.
        """
        if self.tracker is None:
            self.reset_tracker()

        h, w = frame.shape[:2]

        # Detection
        t_det_start = time.perf_counter()
        detections = self.detector.detect(frame)
        detection_latency_ms = (time.perf_counter() - t_det_start) * 1000

        # Tracking
        t_track_start = time.perf_counter()
        tracks = self.tracker.update(detections, (h, w))
        tracking_latency_ms = (time.perf_counter() - t_track_start) * 1000

        return FrameResult(
            frame_num=frame_num,
            detections=detections,
            tracks=tracks,
            detection_latency_ms=detection_latency_ms,
            tracking_latency_ms=tracking_latency_ms
        )

    def process_sequence(
        self,
        frames: Generator[Tuple[int, np.ndarray], None, None]
    ) -> Generator[FrameResult, None, None]:
        """Process a sequence of frames.

        Args:
            frames: Generator yielding (frame_num, frame_image) tuples.

        Yields:
            FrameResult for each frame.
        """
        self.reset_tracker()

        for frame_num, frame in frames:
            yield self.process_frame(frame, frame_num)


class EvaluationPipeline:
    """Full evaluation pipeline for UA-DETRAC."""

    def __init__(
        self,
        config: Config,
        verbose: bool = True
    ):
        """Initialize evaluation pipeline.

        Args:
            config: Configuration instance.
            verbose: Whether to print progress.
        """
        self.config = config
        self.verbose = verbose
        self.pipeline = DetectionTrackingPipeline.from_config(config)
        self.dataset = UADETRACDataset(
            base_path=config.dataset.base_path,
            image_dir=config.dataset.image_dir,
            train_annotation_dir=config.dataset.train_annotation_dir,
            test_annotation_dir=config.dataset.test_annotation_dir,
            image_extension=config.dataset.image_extension
        )

    def _print(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(msg, flush=True)

    def run_evaluation(
        self,
        split: str = "test",
        max_sequences: Optional[int] = None
    ) -> Tuple[Dict[float, DetectionMetrics], TrackingMetrics, float, float, float, int]:
        """Run full evaluation.

        Args:
            split: Dataset split ("train" or "test").
            max_sequences: Maximum number of sequences to evaluate.

        Returns:
            Tuple of (detection_results, tracking_metrics, coco_map,
                      avg_detection_latency_ms, avg_tracking_latency_ms, total_frames).
        """
        all_predictions: Dict[str, Dict[int, List]] = {}
        all_ground_truths: Dict[str, Dict[int, List]] = {}
        all_ignored_regions: Dict[str, List] = {}
        all_detection_latencies: List[float] = []
        all_tracking_latencies: List[float] = []
        total_frames = 0

        sequences = list(self.dataset.iterate_sequences(split, max_sequences))
        self._print(f"Evaluating {len(sequences)} sequences")
        self._print("=" * 70)

        for seq_idx, seq_data in enumerate(sequences):
            self._print(f"\n[{seq_idx+1}/{len(sequences)}] {seq_data.name}")

            # Store GT and ignored regions
            all_ground_truths[seq_data.name] = seq_data.gt_data
            all_ignored_regions[seq_data.name] = seq_data.ignored_regions

            # Process sequence
            predictions: Dict[int, List] = {}
            self.pipeline.reset_tracker()

            frames = self.dataset.iterate_frames(seq_data.name)
            for frame_num, frame in frames:
                result = self.pipeline.process_frame(frame, frame_num)
                predictions[frame_num] = result.get_track_tuples(include_score=True)
                all_detection_latencies.append(result.detection_latency_ms)
                all_tracking_latencies.append(result.tracking_latency_ms)
                total_frames += 1

            all_predictions[seq_data.name] = predictions
            self._print(f"  {len(predictions)} frames processed")

        # Evaluate detection
        self._print("\n" + "=" * 70)
        self._print("Computing detection metrics...")

        det_results = evaluate_detection_with_map(
            all_predictions,
            all_ground_truths,
            all_ignored_regions,
            iou_thresholds=self.config.evaluation.iou_thresholds
        )

        # Compute COCO mAP
        coco_thresholds = np.arange(
            self.config.evaluation.iou_threshold_range_start,
            self.config.evaluation.iou_threshold_range_end + 0.01,
            self.config.evaluation.iou_threshold_range_step
        )
        coco_results = evaluate_detection_with_map(
            all_predictions,
            all_ground_truths,
            all_ignored_regions,
            iou_thresholds=coco_thresholds.tolist()
        )
        coco_map = np.mean([coco_results[t].ap for t in coco_thresholds])

        # Evaluate tracking
        self._print("Computing tracking metrics...")

        track_metrics = evaluate_tracking_multi_sequence(
            all_predictions,
            all_ground_truths,
            all_ignored_regions,
            iou_threshold=self.config.evaluation.tracking_iou_threshold
        )

        avg_detection_latency = np.mean(all_detection_latencies) if all_detection_latencies else 0
        avg_tracking_latency = np.mean(all_tracking_latencies) if all_tracking_latencies else 0

        return det_results, track_metrics, coco_map, avg_detection_latency, avg_tracking_latency, total_frames

    def print_results(
        self,
        det_results: Dict[float, DetectionMetrics],
        track_metrics: TrackingMetrics,
        coco_map: float,
        avg_detection_latency: float,
        avg_tracking_latency: float,
        total_frames: int,
        num_sequences: int
    ) -> None:
        """Print evaluation results.

        Args:
            det_results: Detection results per IoU threshold.
            track_metrics: Tracking metrics.
            coco_map: COCO mAP value.
            avg_detection_latency: Average detection latency in ms.
            avg_tracking_latency: Average tracking latency in ms.
            total_frames: Total number of frames processed.
            num_sequences: Number of sequences evaluated.
        """
        print("\n" + "=" * 70)
        print("DETECTION METRICS")
        print("=" * 70)
        print(f"  Sequences: {num_sequences}")
        print(f"  Frames:    {total_frames}")
        print()

        for iou_thresh, metrics in det_results.items():
            print(f"  [IoU = {iou_thresh}]")
            print(f"    AP:        {metrics.ap*100:.2f}%")
            print(f"    Precision: {metrics.precision*100:.2f}%")
            print(f"    Recall:    {metrics.recall*100:.2f}%")
            print(f"    F1:        {metrics.f1*100:.2f}%")
            print(f"    TP/FP/FN:  {metrics.tp}/{metrics.fp}/{metrics.fn}")
            print()

        print(f"  [COCO mAP (0.5:0.95)]")
        print(f"    mAP:       {coco_map*100:.2f}%")

        print("\n" + "=" * 70)
        print("TRACKING METRICS")
        print("=" * 70)
        print(f"  MOTA:        {track_metrics.mota*100:.2f}%")
        print(f"  MOTP:        {track_metrics.motp:.3f}")
        print(f"  IDF1:        {track_metrics.idf1*100:.2f}%")
        print(f"  ID Switches: {track_metrics.num_switches}")
        print(f"  FP:          {track_metrics.num_false_positives}")
        print(f"  FN:          {track_metrics.num_misses}")

        print("\n" + "=" * 70)
        print("LATENCY")
        print("=" * 70)
        total_latency = avg_detection_latency + avg_tracking_latency
        print(f"  Detection:   {avg_detection_latency:.2f} ms")
        print(f"  Tracking:    {avg_tracking_latency:.2f} ms")
        print(f"  Total:       {total_latency:.2f} ms")
        if total_latency > 0:
            print(f"  FPS:         {1000/total_latency:.1f}")
        print("=" * 70)
