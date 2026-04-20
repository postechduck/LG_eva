"""Tracking evaluation metrics for EVA.

Contains MOTA, MOTP, IDF1 calculation functions using motmetrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import motmetrics as mm
except ImportError:
    mm = None

from ..utils.box import compute_iou, is_in_ignored_region


@dataclass
class TrackingMetrics:
    """Container for tracking evaluation metrics."""
    mota: float
    motp: float
    idf1: float
    num_switches: int
    num_false_positives: int
    num_misses: int
    precision: float
    recall: float
    summary: Optional[pd.DataFrame] = None

    def __str__(self) -> str:
        return (
            f"MOTA: {self.mota*100:.2f}% | "
            f"MOTP: {self.motp:.3f} | "
            f"IDF1: {self.idf1*100:.2f}% | "
            f"ID Sw: {self.num_switches}"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'MOTA': self.mota,
            'MOTP': self.motp,
            'IDF1': self.idf1,
            'ID_Switches': self.num_switches,
            'FP': self.num_false_positives,
            'FN': self.num_misses,
            'Precision': self.precision,
            'Recall': self.recall
        }


def _check_motmetrics():
    """Check if motmetrics is available."""
    if mm is None:
        raise ImportError(
            "motmetrics is required for tracking evaluation. "
            "Install it with: pip install motmetrics"
        )


def evaluate_tracking_single_sequence(
    predictions: Dict[int, List],
    ground_truths: Dict[int, List],
    ignored_regions: Optional[List] = None,
    iou_threshold: float = 0.5
) -> TrackingMetrics:
    """Evaluate tracking for a single sequence.

    Args:
        predictions: Dict mapping frame_num -> [(id, x1, y1, x2, y2, ...), ...].
        ground_truths: Dict mapping frame_num -> [(id, x1, y1, x2, y2), ...].
        ignored_regions: List of ignored regions [(x1, y1, x2, y2), ...].
        iou_threshold: IoU threshold for matching.

    Returns:
        TrackingMetrics object.
    """
    _check_motmetrics()

    acc = mm.MOTAccumulator(auto_id=True)
    ignored_regions = ignored_regions or []

    for frame_num in sorted(predictions.keys()):
        gt_boxes = ground_truths.get(frame_num, [])
        pred_boxes = predictions.get(frame_num, [])

        # Filter valid GT and predictions (not in ignored regions)
        valid_gt = [
            gt for gt in gt_boxes
            if not is_in_ignored_region(gt[1:5], ignored_regions)
        ]
        valid_pred = [
            pred for pred in pred_boxes
            if not is_in_ignored_region(pred[1:5], ignored_regions)
        ]

        gt_ids = [gt[0] for gt in valid_gt]
        gt_bboxes = [gt[1:5] for gt in valid_gt]
        pred_ids = [pred[0] for pred in valid_pred]
        pred_bboxes = [pred[1:5] for pred in valid_pred]

        # Calculate distance matrix (1 - IoU)
        if len(gt_bboxes) > 0 and len(pred_bboxes) > 0:
            distances = np.zeros((len(gt_bboxes), len(pred_bboxes)))
            for i, gt_box in enumerate(gt_bboxes):
                for j, pred_box in enumerate(pred_bboxes):
                    iou = compute_iou(gt_box, pred_box)
                    distances[i, j] = 1 - iou
            # Set distances above threshold to NaN (unmatched)
            distances[distances > iou_threshold] = np.nan
        else:
            distances = np.empty((len(gt_bboxes), len(pred_bboxes)))
            distances[:] = np.nan

        acc.update(gt_ids, pred_ids, distances)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'mota', 'motp', 'idf1', 'num_switches',
        'num_false_positives', 'num_misses',
        'precision', 'recall'
    ], name='eval')

    return TrackingMetrics(
        mota=float(summary.iloc[0]['mota']),
        motp=float(summary.iloc[0]['motp']),
        idf1=float(summary.iloc[0]['idf1']),
        num_switches=int(summary.iloc[0]['num_switches']),
        num_false_positives=int(summary.iloc[0]['num_false_positives']),
        num_misses=int(summary.iloc[0]['num_misses']),
        precision=float(summary.iloc[0]['precision']),
        recall=float(summary.iloc[0]['recall']),
        summary=summary
    )


def evaluate_tracking_multi_sequence(
    all_predictions: Dict[str, Dict[int, List]],
    all_ground_truths: Dict[str, Dict[int, List]],
    all_ignored_regions: Dict[str, List],
    iou_threshold: float = 0.5
) -> TrackingMetrics:
    """Evaluate tracking across multiple sequences.

    Args:
        all_predictions: Dict mapping seq_name -> {frame_num: [(id, x1, y1, x2, y2, ...), ...]}.
        all_ground_truths: Dict mapping seq_name -> {frame_num: [(id, x1, y1, x2, y2), ...]}.
        all_ignored_regions: Dict mapping seq_name -> [(x1, y1, x2, y2), ...].
        iou_threshold: IoU threshold for matching.

    Returns:
        TrackingMetrics object with aggregate metrics.
    """
    _check_motmetrics()

    accumulators = []
    names = []

    for seq_name in all_predictions.keys():
        acc = mm.MOTAccumulator(auto_id=True)
        predictions = all_predictions[seq_name]
        ground_truths = all_ground_truths.get(seq_name, {})
        ignored_regions = all_ignored_regions.get(seq_name, [])

        for frame_num in sorted(predictions.keys()):
            gt_boxes = ground_truths.get(frame_num, [])
            pred_boxes = predictions.get(frame_num, [])

            # Filter valid GT and predictions
            valid_gt = [
                gt for gt in gt_boxes
                if not is_in_ignored_region(gt[1:5], ignored_regions)
            ]
            valid_pred = [
                pred for pred in pred_boxes
                if not is_in_ignored_region(pred[1:5], ignored_regions)
            ]

            gt_ids = [gt[0] for gt in valid_gt]
            gt_bboxes = [gt[1:5] for gt in valid_gt]
            pred_ids = [pred[0] for pred in valid_pred]
            pred_bboxes = [pred[1:5] for pred in valid_pred]

            # Calculate distance matrix
            if len(gt_bboxes) > 0 and len(pred_bboxes) > 0:
                distances = np.zeros((len(gt_bboxes), len(pred_bboxes)))
                for i, gt_box in enumerate(gt_bboxes):
                    for j, pred_box in enumerate(pred_bboxes):
                        iou = compute_iou(gt_box, pred_box)
                        distances[i, j] = 1 - iou
                distances[distances > iou_threshold] = np.nan
            else:
                distances = np.empty((len(gt_bboxes), len(pred_bboxes)))
                distances[:] = np.nan

            acc.update(gt_ids, pred_ids, distances)

        accumulators.append(acc)
        names.append(seq_name)

    # Compute aggregate metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(accumulators, names=names, metrics=[
        'mota', 'motp', 'idf1', 'num_switches',
        'num_false_positives', 'num_misses',
        'precision', 'recall'
    ], generate_overall=True)

    # Get overall metrics (last row)
    overall = summary.loc['OVERALL']

    return TrackingMetrics(
        mota=float(overall['mota']),
        motp=float(overall['motp']),
        idf1=float(overall['idf1']),
        num_switches=int(overall['num_switches']),
        num_false_positives=int(overall['num_false_positives']),
        num_misses=int(overall['num_misses']),
        precision=float(overall['precision']),
        recall=float(overall['recall']),
        summary=summary
    )
