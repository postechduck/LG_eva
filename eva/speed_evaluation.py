"""Speed evaluation - compare predicted speed with GT speed.

Compares pixel speed calculated from tracking results vs ground truth.
Handles TP, FP, FN cases separately.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from .speed import (
    calculate_center, SpeedClass, classify_speed_value,
    THRESHOLD_UNIFORM, THRESHOLD_SEMANTIC
)
from .utils.box import compute_iou


@dataclass
class SpeedComparisonResult:
    """Result of speed comparison for a single frame/track."""
    frame_num: int
    gt_track_id: int
    pred_track_id: int
    gt_speed: float
    pred_speed: float
    speed_error: float  # pred - gt
    gt_class: SpeedClass
    pred_class: SpeedClass
    class_match: bool
    iou: float


@dataclass
class SpeedEvaluationMetrics:
    """Overall speed evaluation metrics."""
    # Sample counts
    num_tp_samples: int  # Matched samples (for speed comparison)
    num_fn_samples: int  # GT samples not matched (missed)
    num_fp_samples: int  # Pred samples not matched (false alarm)

    # Coverage metrics
    gt_coverage: float   # % of GT speeds that were matched
    pred_precision: float  # % of Pred speeds that have GT match

    # Speed accuracy (TP only)
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    correlation: float  # Pearson correlation
    class_accuracy: float  # Speed class match rate

    # Per-class metrics
    low_accuracy: float
    medium_accuracy: float
    high_accuracy: float

    # Confusion matrix counts
    confusion_matrix: Dict[str, Dict[str, int]]

    def __str__(self) -> str:
        return (
            f"Speed Evaluation Metrics:\n"
            f"  Coverage:\n"
            f"    GT Coverage: {self.gt_coverage*100:.1f}% ({self.num_tp_samples}/{self.num_tp_samples + self.num_fn_samples})\n"
            f"    Pred Precision: {self.pred_precision*100:.1f}% ({self.num_tp_samples}/{self.num_tp_samples + self.num_fp_samples})\n"
            f"    FN (missed GT): {self.num_fn_samples}\n"
            f"    FP (false pred): {self.num_fp_samples}\n"
            f"  Speed Accuracy (TP only):\n"
            f"    Samples: {self.num_tp_samples}\n"
            f"    MAE: {self.mae:.3f} px/frame\n"
            f"    RMSE: {self.rmse:.3f} px/frame\n"
            f"    Correlation: {self.correlation:.3f}\n"
            f"    Class Accuracy: {self.class_accuracy*100:.1f}%\n"
            f"      - LOW: {self.low_accuracy*100:.1f}%\n"
            f"      - MEDIUM: {self.medium_accuracy*100:.1f}%\n"
            f"      - HIGH: {self.high_accuracy*100:.1f}%"
        )


def calculate_gt_speeds(
    gt_data: Dict[int, List[Tuple]],
    ignored_regions: List[Tuple] = None
) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    """Calculate pixel speeds from ground truth data.

    Args:
        gt_data: Dict mapping frame_num -> [(track_id, x1, y1, x2, y2), ...].
        ignored_regions: List of ignored regions (optional).

    Returns:
        Dict mapping track_id -> {frame_num: (center_x, center_y, speed)}.
    """
    from .utils.box import is_in_ignored_region

    ignored_regions = ignored_regions or []

    # Reorganize by track_id
    track_frames: Dict[int, List[Tuple[int, float, float]]] = {}

    for frame_num, targets in gt_data.items():
        for target in targets:
            track_id = target[0]
            box = target[1:5]

            # Skip if in ignored region
            if ignored_regions and is_in_ignored_region(box, ignored_regions):
                continue

            cx, cy = calculate_center(box)

            if track_id not in track_frames:
                track_frames[track_id] = []
            track_frames[track_id].append((frame_num, cx, cy))

    # Calculate speeds
    gt_speeds: Dict[int, Dict[int, Tuple[float, float, float]]] = {}

    for track_id, frames in track_frames.items():
        frames.sort(key=lambda x: x[0])
        gt_speeds[track_id] = {}

        for i, (frame_num, cx, cy) in enumerate(frames):
            if i == 0:
                speed = 0.0
            else:
                prev_frame, prev_cx, prev_cy = frames[i-1]
                frame_gap = frame_num - prev_frame
                if frame_gap > 0:
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    speed = math.sqrt(dx*dx + dy*dy) / frame_gap
                else:
                    speed = 0.0

            gt_speeds[track_id][frame_num] = (cx, cy, speed)

    return gt_speeds


def match_tracks_per_frame(
    predictions: Dict[int, List[Tuple]],
    gt_data: Dict[int, List[Tuple]],
    iou_threshold: float = 0.5
) -> Tuple[Dict[int, List[Tuple[int, int, float]]], Dict[int, List[int]], Dict[int, List[int]]]:
    """Match predicted tracks to GT tracks per frame based on IoU.

    Args:
        predictions: Predicted tracks per frame.
        gt_data: GT tracks per frame.
        iou_threshold: Minimum IoU for matching.

    Returns:
        Tuple of:
        - matches: Dict mapping frame_num -> [(gt_track_id, pred_track_id, iou), ...]
        - fn_per_frame: Dict mapping frame_num -> [unmatched gt_track_ids]
        - fp_per_frame: Dict mapping frame_num -> [unmatched pred_track_ids]
    """
    matches: Dict[int, List[Tuple[int, int, float]]] = {}
    fn_per_frame: Dict[int, List[int]] = {}  # False Negatives (missed GT)
    fp_per_frame: Dict[int, List[int]] = {}  # False Positives (wrong pred)

    all_frames = set(predictions.keys()) | set(gt_data.keys())

    for frame_num in all_frames:
        pred_tracks = predictions.get(frame_num, [])
        gt_tracks = gt_data.get(frame_num, [])

        frame_matches = []
        gt_matched = set()
        pred_matched = set()

        # For each prediction, find best matching GT
        for pred in pred_tracks:
            pred_id = pred[0]
            pred_box = pred[1:5]

            best_iou = 0
            best_gt_id = None

            for gt in gt_tracks:
                gt_id = gt[0]
                if gt_id in gt_matched:
                    continue

                gt_box = gt[1:5]
                iou = compute_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_id = gt_id

            if best_iou >= iou_threshold and best_gt_id is not None:
                frame_matches.append((best_gt_id, pred_id, best_iou))
                gt_matched.add(best_gt_id)
                pred_matched.add(pred_id)

        # Record matches
        if frame_matches:
            matches[frame_num] = frame_matches

        # Record FN (GT not matched)
        fn_ids = [gt[0] for gt in gt_tracks if gt[0] not in gt_matched]
        if fn_ids:
            fn_per_frame[frame_num] = fn_ids

        # Record FP (Pred not matched)
        fp_ids = [pred[0] for pred in pred_tracks if pred[0] not in pred_matched]
        if fp_ids:
            fp_per_frame[frame_num] = fp_ids

    return matches, fn_per_frame, fp_per_frame


def evaluate_speed_accuracy(
    predictions: Dict[int, List[Tuple]],
    gt_data: Dict[int, List[Tuple]],
    ignored_regions: List[Tuple] = None,
    low_threshold: float = 1.0,
    high_threshold: float = 5.0,
    iou_threshold: float = 0.5
) -> Tuple[SpeedEvaluationMetrics, List[SpeedComparisonResult]]:
    """Evaluate speed prediction accuracy against ground truth.

    Args:
        predictions: Predicted tracks per frame.
        gt_data: GT tracks per frame.
        ignored_regions: Ignored regions.
        low_threshold: Speed threshold for LOW class.
        high_threshold: Speed threshold for HIGH class.
        iou_threshold: IoU threshold for matching.

    Returns:
        Tuple of (SpeedEvaluationMetrics, list of per-sample results).
    """
    from .speed import calculate_pixel_speed
    from .utils.box import is_in_ignored_region

    ignored_regions = ignored_regions or []

    # Calculate speeds
    gt_speeds = calculate_gt_speeds(gt_data, ignored_regions)
    pred_speed_results = calculate_pixel_speed(predictions)

    # Convert pred results to same format
    pred_speeds: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
    for track_id, results in pred_speed_results.items():
        pred_speeds[track_id] = {}
        for r in results:
            pred_speeds[track_id][r.frame_num] = (r.center_x, r.center_y, r.pixel_speed)

    # Match tracks per frame (with FN/FP tracking)
    matches, fn_per_frame, fp_per_frame = match_tracks_per_frame(
        predictions, gt_data, iou_threshold
    )

    # Count FN samples (GT speeds not matched)
    num_fn_samples = 0
    for frame_num, fn_ids in fn_per_frame.items():
        for gt_id in fn_ids:
            if gt_id in gt_speeds and frame_num in gt_speeds[gt_id]:
                _, _, speed = gt_speeds[gt_id][frame_num]
                if speed > 0:  # Skip first frame
                    num_fn_samples += 1

    # Count FP samples (Pred speeds not matched to any GT)
    num_fp_samples = 0
    for frame_num, fp_ids in fp_per_frame.items():
        for pred_id in fp_ids:
            if pred_id in pred_speeds and frame_num in pred_speeds[pred_id]:
                # Check if in ignored region
                cx, cy, speed = pred_speeds[pred_id][frame_num]
                if speed > 0:  # Skip first frame
                    num_fp_samples += 1

    # Compare speeds for matched pairs (TP)
    comparisons: List[SpeedComparisonResult] = []
    gt_speed_list = []
    pred_speed_list = []

    for frame_num, frame_matches in matches.items():
        for gt_id, pred_id, iou in frame_matches:
            # Get speeds
            if gt_id not in gt_speeds or frame_num not in gt_speeds[gt_id]:
                continue
            if pred_id not in pred_speeds or frame_num not in pred_speeds[pred_id]:
                continue

            _, _, gt_speed = gt_speeds[gt_id][frame_num]
            _, _, pred_speed = pred_speeds[pred_id][frame_num]

            # Skip first frames (speed = 0)
            if gt_speed == 0 and pred_speed == 0:
                continue

            # Classify
            gt_class = classify_speed_value(gt_speed, low_threshold, high_threshold)
            pred_class = classify_speed_value(pred_speed, low_threshold, high_threshold)

            comparison = SpeedComparisonResult(
                frame_num=frame_num,
                gt_track_id=gt_id,
                pred_track_id=pred_id,
                gt_speed=gt_speed,
                pred_speed=pred_speed,
                speed_error=pred_speed - gt_speed,
                gt_class=gt_class,
                pred_class=pred_class,
                class_match=(gt_class == pred_class),
                iou=iou
            )
            comparisons.append(comparison)

            gt_speed_list.append(gt_speed)
            pred_speed_list.append(pred_speed)

    num_tp_samples = len(comparisons)

    if num_tp_samples == 0:
        return None, []

    # Calculate metrics
    gt_arr = np.array(gt_speed_list)
    pred_arr = np.array(pred_speed_list)

    errors = pred_arr - gt_arr
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # Correlation
    if len(gt_arr) > 1 and np.std(gt_arr) > 0 and np.std(pred_arr) > 0:
        correlation = np.corrcoef(gt_arr, pred_arr)[0, 1]
    else:
        correlation = 0.0

    # Class accuracy
    class_matches = sum(1 for c in comparisons if c.class_match)
    class_accuracy = class_matches / len(comparisons)

    # Per-class accuracy
    confusion = {
        'LOW': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
        'MEDIUM': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
        'HIGH': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
    }

    for c in comparisons:
        gt_name = c.gt_class.name
        pred_name = c.pred_class.name
        confusion[gt_name][pred_name] += 1

    def calc_class_acc(class_name):
        total = sum(confusion[class_name].values())
        if total == 0:
            return 0.0
        return confusion[class_name][class_name] / total

    # Coverage metrics
    total_gt_samples = num_tp_samples + num_fn_samples
    total_pred_samples = num_tp_samples + num_fp_samples

    gt_coverage = num_tp_samples / total_gt_samples if total_gt_samples > 0 else 0
    pred_precision = num_tp_samples / total_pred_samples if total_pred_samples > 0 else 0

    metrics = SpeedEvaluationMetrics(
        num_tp_samples=num_tp_samples,
        num_fn_samples=num_fn_samples,
        num_fp_samples=num_fp_samples,
        gt_coverage=gt_coverage,
        pred_precision=pred_precision,
        mae=mae,
        rmse=rmse,
        correlation=correlation,
        class_accuracy=class_accuracy,
        low_accuracy=calc_class_acc('LOW'),
        medium_accuracy=calc_class_acc('MEDIUM'),
        high_accuracy=calc_class_acc('HIGH'),
        confusion_matrix=confusion
    )

    return metrics, comparisons


def print_speed_evaluation_summary(metrics: SpeedEvaluationMetrics) -> None:
    """Print comprehensive speed evaluation summary."""

    total_gt = metrics.num_tp_samples + metrics.num_fn_samples
    total_pred = metrics.num_tp_samples + metrics.num_fp_samples

    print("\n" + "=" * 60)
    print("SPEED EVALUATION SUMMARY")
    print("=" * 60)

    print("\n[1] Coverage Analysis")
    print("-" * 40)
    print(f"  GT 총 샘플:        {total_gt:,}")
    print(f"    - TP (매칭됨):   {metrics.num_tp_samples:,} ({metrics.gt_coverage*100:.1f}%)")
    print(f"    - FN (놓침):     {metrics.num_fn_samples:,} ({(1-metrics.gt_coverage)*100:.1f}%)")
    print()
    print(f"  Pred 총 샘플:      {total_pred:,}")
    print(f"    - TP (정확):     {metrics.num_tp_samples:,} ({metrics.pred_precision*100:.1f}%)")
    print(f"    - FP (오탐):     {metrics.num_fp_samples:,} ({(1-metrics.pred_precision)*100:.1f}%)")

    print("\n[2] Speed Accuracy (TP만 대상)")
    print("-" * 40)
    print(f"  MAE:          {metrics.mae:.3f} px/frame")
    print(f"  RMSE:         {metrics.rmse:.3f} px/frame")
    print(f"  Correlation:  {metrics.correlation:.3f}")

    print("\n[3] Speed Classification (TP만 대상)")
    print("-" * 40)
    print(f"  Class Accuracy: {metrics.class_accuracy*100:.1f}%")
    print(f"    - LOW:    {metrics.low_accuracy*100:.1f}%")
    print(f"    - MEDIUM: {metrics.medium_accuracy*100:.1f}%")
    print(f"    - HIGH:   {metrics.high_accuracy*100:.1f}%")

    print("\n[4] FN/FP 영향 분석")
    print("-" * 40)
    print(f"  FN 영향: {metrics.num_fn_samples:,}개 차량의 속도를 추정하지 못함")
    print(f"  FP 영향: {metrics.num_fp_samples:,}개의 잘못된 속도 추정 발생")


def print_confusion_matrix(confusion: Dict[str, Dict[str, int]]) -> None:
    """Print confusion matrix in a readable format."""
    print("\nConfusion Matrix (rows=GT, cols=Pred):")
    print("-" * 45)
    print(f"{'GT \\ Pred':<12} {'LOW':>10} {'MEDIUM':>10} {'HIGH':>10}")
    print("-" * 45)

    for gt_class in ['LOW', 'MEDIUM', 'HIGH']:
        row = confusion[gt_class]
        total = sum(row.values())
        print(f"{gt_class:<12} {row['LOW']:>10} {row['MEDIUM']:>10} {row['HIGH']:>10}  (n={total})")

    print("-" * 45)
