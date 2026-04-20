"""Detection evaluation metrics for EVA.

Contains AP, mAP, Precision, Recall, F1 calculation functions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.box import compute_iou, is_in_ignored_region


@dataclass
class DetectionMetrics:
    """Container for detection evaluation metrics."""
    ap: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    total_gt: int
    iou_threshold: float

    def __str__(self) -> str:
        return (
            f"AP@{self.iou_threshold}: {self.ap*100:.2f}% | "
            f"P: {self.precision*100:.2f}% | "
            f"R: {self.recall*100:.2f}% | "
            f"F1: {self.f1*100:.2f}%"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'AP': self.ap,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1': self.f1,
            'TP': self.tp,
            'FP': self.fp,
            'FN': self.fn,
            'Total_GT': self.total_gt,
            'IoU_Threshold': self.iou_threshold
        }


def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """Compute Average Precision using 11-point interpolation.

    Args:
        precisions: List of precision values.
        recalls: List of recall values.

    Returns:
        AP value between 0 and 1.
    """
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_above_t = [
            p for p, r in zip(precisions, recalls) if r >= t
        ]
        if precisions_above_t:
            ap += max(precisions_above_t)
    return ap / 11


def evaluate_detection_with_map(
    all_predictions: Dict[str, Dict[int, List]],
    all_ground_truths: Dict[str, Dict[int, List]],
    all_ignored_regions: Dict[str, List],
    iou_thresholds: List[float] = [0.5]
) -> Dict[float, DetectionMetrics]:
    """Evaluate detection performance with mAP calculation.

    Args:
        all_predictions: Dict mapping seq_name -> {frame_num: [(id, x1, y1, x2, y2, conf), ...]}.
        all_ground_truths: Dict mapping seq_name -> {frame_num: [(id, x1, y1, x2, y2), ...]}.
        all_ignored_regions: Dict mapping seq_name -> [(x1, y1, x2, y2), ...].
        iou_thresholds: List of IoU thresholds to evaluate.

    Returns:
        Dict mapping IoU threshold to DetectionMetrics.
    """
    results = {}

    for iou_thresh in iou_thresholds:
        all_detections = []  # [(confidence, is_tp), ...]
        total_gt = 0

        for seq_name in all_predictions.keys():
            predictions = all_predictions[seq_name]
            ground_truths = all_ground_truths.get(seq_name, {})
            ignored_regions = all_ignored_regions.get(seq_name, [])

            for frame_num in predictions.keys():
                gt_boxes = ground_truths.get(frame_num, [])
                pred_boxes = predictions.get(frame_num, [])

                # Filter valid GT (not in ignored region)
                valid_gt = [
                    gt for gt in gt_boxes
                    if not is_in_ignored_region(gt[1:5], ignored_regions)
                ]
                total_gt += len(valid_gt)

                gt_matched = [False] * len(valid_gt)

                for pred in pred_boxes:
                    pred_box = pred[1:5]
                    conf = pred[5] if len(pred) > 5 else 1.0

                    # Skip if in ignored region
                    if is_in_ignored_region(pred_box, ignored_regions):
                        continue

                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx, gt in enumerate(valid_gt):
                        if gt_matched[gt_idx]:
                            continue
                        gt_box = gt[1:5]
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        gt_matched[best_gt_idx] = True
                        all_detections.append((conf, True))
                    else:
                        all_detections.append((conf, False))

        # Sort by confidence (descending)
        all_detections.sort(key=lambda x: x[0], reverse=True)

        # Calculate precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for conf, is_tp in all_detections:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt if total_gt > 0 else 0
            precisions.append(precision)
            recalls.append(recall)

        # Calculate AP
        ap = compute_ap(precisions, recalls)

        # Final metrics
        final_tp = tp_cumsum
        final_fp = fp_cumsum
        final_fn = total_gt - final_tp

        final_precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0
        final_recall = final_tp / total_gt if total_gt > 0 else 0
        final_f1 = (
            2 * final_precision * final_recall / (final_precision + final_recall)
            if (final_precision + final_recall) > 0 else 0
        )

        results[iou_thresh] = DetectionMetrics(
            ap=ap,
            precision=final_precision,
            recall=final_recall,
            f1=final_f1,
            tp=final_tp,
            fp=final_fp,
            fn=final_fn,
            total_gt=total_gt,
            iou_threshold=iou_thresh
        )

    return results


def evaluate_detection_simple(
    predictions: Dict[int, List],
    ground_truths: Dict[int, List],
    evaluated_frames: List[int],
    iou_threshold: float = 0.5
) -> DetectionMetrics:
    """Simple detection evaluation without mAP (for single sequence).

    Args:
        predictions: Dict mapping frame_num -> [(id, x1, y1, x2, y2), ...].
        ground_truths: Dict mapping frame_num -> [(id, x1, y1, x2, y2), ...].
        evaluated_frames: List of frame numbers to evaluate.
        iou_threshold: IoU threshold for matching.

    Returns:
        DetectionMetrics object.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    for frame_num in evaluated_frames:
        gt_boxes = ground_truths.get(frame_num, [])
        pred_boxes = predictions.get(frame_num, [])
        gt_matched = [False] * len(gt_boxes)

        for pred in pred_boxes:
            pred_box = pred[1:5]
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                gt_box = gt[1:5]
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                total_tp += 1
                gt_matched[best_gt_idx] = True
                all_ious.append(best_iou)
            else:
                total_fp += 1

        total_fn += sum(1 for m in gt_matched if not m)

    total_gt = total_tp + total_fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # For simple evaluation, AP is approximated by precision
    return DetectionMetrics(
        ap=precision,  # Simplified: using precision as proxy
        precision=precision,
        recall=recall,
        f1=f1,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        total_gt=total_gt,
        iou_threshold=iou_threshold
    )


def compute_coco_map(
    all_predictions: Dict[str, Dict[int, List]],
    all_ground_truths: Dict[str, Dict[int, List]],
    all_ignored_regions: Dict[str, List],
    iou_start: float = 0.5,
    iou_end: float = 0.95,
    iou_step: float = 0.05
) -> float:
    """Compute COCO-style mAP@0.5:0.95.

    Args:
        all_predictions: Predictions per sequence.
        all_ground_truths: Ground truths per sequence.
        all_ignored_regions: Ignored regions per sequence.
        iou_start: Starting IoU threshold.
        iou_end: Ending IoU threshold.
        iou_step: IoU threshold step size.

    Returns:
        mAP value.
    """
    thresholds = np.arange(iou_start, iou_end + iou_step/2, iou_step)
    results = evaluate_detection_with_map(
        all_predictions, all_ground_truths, all_ignored_regions,
        iou_thresholds=thresholds.tolist()
    )
    return np.mean([results[t].ap for t in thresholds])
