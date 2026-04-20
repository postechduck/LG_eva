#!/usr/bin/env python3
"""
Compare detection models (RT-DETR vs YOLOv8x).

Usage:
    python scripts/compare.py
    python scripts/compare.py --max_sequences 5
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eva import (
    Config,
    UADETRACDataset,
    create_detector,
    evaluate_detection_with_map,
    compute_iou,
    is_in_ignored_region,
)
from eva.data import parse_ua_detrac_xml


def evaluate_detector(
    detector,
    detector_name: str,
    dataset: UADETRACDataset,
    max_sequences: int = None,
    conf_threshold: float = 0.3
) -> Dict:
    """Evaluate a single detector on the test set."""

    sequences = dataset.get_test_sequences()
    if max_sequences:
        sequences = sequences[:max_sequences]

    all_detections = []
    total_gt = 0
    all_latencies = []
    total_frames = 0

    for seq_idx, seq_name in enumerate(sequences):
        print(f"  [{seq_idx+1}/{len(sequences)}] {seq_name}", end="", flush=True)

        seq_data = dataset.get_sequence_data(seq_name, split="test")
        if seq_data is None:
            print(" - skipped")
            continue

        seq_start = time.time()

        for frame_num, frame in dataset.iterate_frames(seq_name):
            # Detection with timing
            t_start = time.perf_counter()
            detections = detector.detect(frame)
            t_end = time.perf_counter()
            all_latencies.append((t_end - t_start) * 1000)

            # GT processing
            gt_boxes = seq_data.gt_data.get(frame_num, [])
            valid_gt = [
                gt[1:5] for gt in gt_boxes
                if not is_in_ignored_region(gt[1:5], seq_data.ignored_regions)
            ]
            total_gt += len(valid_gt)

            gt_matched = [False] * len(valid_gt)

            # Match detections to GT
            for det in detections:
                pred_box = det.box

                if is_in_ignored_region(pred_box, seq_data.ignored_regions):
                    continue

                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt_box in enumerate(valid_gt):
                    if gt_matched[gt_idx]:
                        continue
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= 0.5 and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    all_detections.append((det.confidence, True))
                else:
                    all_detections.append((det.confidence, False))

            total_frames += 1

        seq_time = time.time() - seq_start
        print(f" ({seq_data.num_frames} frames, {seq_time:.1f}s)", flush=True)

    # Calculate AP
    all_detections.sort(key=lambda x: x[0], reverse=True)

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

    # 11-point AP
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        precisions_above_t = [p for p, r in zip(precisions, recalls) if r >= t]
        if precisions_above_t:
            ap += max(precisions_above_t)
    ap = ap / 11

    final_tp = tp_cumsum
    final_fp = fp_cumsum
    final_fn = total_gt - final_tp

    final_precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0
    final_recall = final_tp / total_gt if total_gt > 0 else 0
    final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

    avg_latency = np.mean(all_latencies)

    return {
        'AP': ap,
        'Precision': final_precision,
        'Recall': final_recall,
        'F1': final_f1,
        'TP': final_tp,
        'FP': final_fp,
        'FN': final_fn,
        'Total_GT': total_gt,
        'Latency_ms': avg_latency,
        'FPS': 1000 / avg_latency if avg_latency > 0 else 0,
        'Total_Frames': total_frames
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare detection models (RT-DETR vs YOLOv8x)'
    )
    parser.add_argument(
        '--max_sequences', type=int, default=None,
        help='Maximum number of sequences to evaluate'
    )
    parser.add_argument(
        '--conf', type=float, default=0.3,
        help='Confidence threshold'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Detection Model Comparison: RT-DETR vs YOLOv8x")
    print("=" * 70)

    config = Config()
    dataset = UADETRACDataset(base_path=config.dataset.base_path)

    print(f"\nDataset: UA-DETRAC Test Set")
    print(f"Sequences: {len(dataset.get_test_sequences())}" +
          (f" (using first {args.max_sequences})" if args.max_sequences else ""))
    print(f"Confidence Threshold: {args.conf}")
    print(f"IoU Threshold: 0.5")
    print("=" * 70)

    results = {}

    # RT-DETR evaluation
    print("\n[1] RT-DETR-L Evaluation")
    rtdetr = create_detector(
        model_type="rtdetr",
        model_path=config.detection.model_path,
        confidence_threshold=args.conf
    )
    rtdetr.warmup()
    results['RT-DETR-L'] = evaluate_detector(
        rtdetr, 'RT-DETR-L', dataset,
        max_sequences=args.max_sequences,
        conf_threshold=args.conf
    )
    del rtdetr

    # YOLOv8x evaluation
    print("\n[2] YOLOv8x Evaluation")
    yolo_path = str(Path(config.detection.model_path).parent / 'yolov8x.pt')
    yolo = create_detector(
        model_type="yolo",
        model_path=yolo_path,
        confidence_threshold=args.conf
    )
    yolo.warmup()
    results['YOLOv8x'] = evaluate_detector(
        yolo, 'YOLOv8x', dataset,
        max_sequences=args.max_sequences,
        conf_threshold=args.conf
    )
    del yolo

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n### Detection Performance (IoU=0.5)")
    print("-" * 70)
    print(f"{'Model':<15} {'AP':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['AP']*100:>6.2f}%   "
              f"{metrics['Precision']*100:>8.2f}%   "
              f"{metrics['Recall']*100:>6.2f}%   "
              f"{metrics['F1']*100:>6.2f}%")

    print("\n### TP / FP / FN")
    print("-" * 70)
    print(f"{'Model':<15} {'TP':<12} {'FP':<12} {'FN':<12} {'Total GT':<12}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['TP']:<12} {metrics['FP']:<12} "
              f"{metrics['FN']:<12} {metrics['Total_GT']:<12}")

    print("\n### Latency")
    print("-" * 70)
    print(f"{'Model':<15} {'Latency (ms)':<15} {'FPS':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['Latency_ms']:>10.2f}     "
              f"{metrics['FPS']:>6.1f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
