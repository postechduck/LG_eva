#!/usr/bin/env python3
"""
Full evaluation script - Detection, Tracking, and Speed evaluation.

Combines all evaluation metrics into a comprehensive report.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eva import (
    Config, DetectionTrackingPipeline, UADETRACDataset,
    evaluate_detection_with_map, evaluate_tracking_multi_sequence,
    calculate_pixel_speed, analyze_speed_distribution,
    THRESHOLD_SEMANTIC
)
from eva.speed_evaluation import (
    evaluate_speed_accuracy, print_speed_evaluation_summary,
    print_confusion_matrix, calculate_gt_speeds
)


def run_full_evaluation(
    config: Config,
    max_sequences: int = None,
    split: str = 'test',
    verbose: bool = True
):
    """Run full evaluation pipeline."""

    pipeline = DetectionTrackingPipeline.from_config(config)
    dataset = UADETRACDataset(config.dataset.base_path)

    sequences = (
        dataset.get_test_sequences() if split == 'test'
        else dataset.get_train_sequences()
    )

    if max_sequences:
        sequences = sequences[:max_sequences]

    print("=" * 70)
    print("FULL EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Model: {config.detection.model_type.upper()}")
    print(f"Confidence Threshold: {config.detection.confidence_threshold}")
    print(f"Sequences: {len(sequences)}")
    print("=" * 70)

    # Storage for all results
    all_predictions = {}  # seq_name -> {frame_num -> tracks}
    all_gt_data = {}      # seq_name -> {frame_num -> targets}
    all_ignored = {}      # seq_name -> ignored_regions

    # Timing
    total_det_time = 0
    total_track_time = 0
    total_frames = 0

    # Process all sequences
    for seq_idx, seq_name in enumerate(sequences):
        if verbose:
            print(f"\n[{seq_idx+1}/{len(sequences)}] Processing {seq_name}...")

        # Load GT using get_sequence_data
        seq_data = dataset.get_sequence_data(seq_name, split)
        if seq_data is None:
            print(f"  Warning: Could not load sequence {seq_name}")
            continue

        gt_data = seq_data.gt_data
        ignored_regions = seq_data.ignored_regions

        all_gt_data[seq_name] = gt_data
        all_ignored[seq_name] = ignored_regions

        # Process frames
        predictions = {}
        pipeline.reset_tracker()

        for frame_num, frame in dataset.iterate_frames(seq_name):
            result = pipeline.process_frame(frame, frame_num)
            predictions[frame_num] = result.get_track_tuples(include_score=True)

            total_det_time += result.detection_latency_ms
            total_track_time += result.tracking_latency_ms
            total_frames += 1

        all_predictions[seq_name] = predictions

        if verbose:
            print(f"  Frames: {len(predictions)}, GT targets: {sum(len(v) for v in gt_data.values())}")

    # =========================================================================
    # 1. DETECTION EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. DETECTION EVALUATION")
    print("=" * 70)

    print("\n[평가 방법]")
    print("  - IoU 기반 매칭: Prediction과 GT를 IoU로 매칭")
    print("  - AP (Average Precision): Precision-Recall 곡선의 면적")
    print("  - Confidence 기반 정렬: 높은 confidence부터 순차적으로 매칭")
    print("  - Ignored Region: 해당 영역 내 GT/Pred는 평가에서 제외")

    # Calculate detection metrics
    # evaluate_detection_with_map expects: Dict[seq_name -> Dict[frame_num -> List]]
    det_results = evaluate_detection_with_map(
        all_predictions, all_gt_data, all_ignored,
        iou_thresholds=[0.5, 0.75]
    )

    det_metrics = det_results[0.5]  # Get metrics at IoU=0.5

    # Calculate COCO mAP
    from eva.evaluation.detection import compute_coco_map
    coco_map = compute_coco_map(all_predictions, all_gt_data, all_ignored)

    det_metrics_50 = det_results[0.5]
    det_metrics_75 = det_results[0.75]

    # Count totals
    total_preds = sum(sum(len(tracks) for tracks in seq.values()) for seq in all_predictions.values())
    total_gt = sum(sum(len(targets) for targets in seq.values()) for seq in all_gt_data.values())

    print("\n[Detection 결과]")
    print("-" * 50)
    print(f"  Total Predictions: {total_preds:,}")
    print(f"  Total GT Boxes:    {total_gt:,}")
    print()
    print(f"  AP@0.5:      {det_metrics_50.ap*100:.2f}%")
    print(f"  AP@0.75:     {det_metrics_75.ap*100:.2f}%")
    print(f"  COCO mAP:    {coco_map*100:.2f}%")
    print()
    print(f"  Precision:   {det_metrics_50.precision*100:.2f}%")
    print(f"  Recall:      {det_metrics_50.recall*100:.2f}%")
    print(f"  F1 Score:    {det_metrics_50.f1*100:.2f}%")

    # =========================================================================
    # 2. TRACKING EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. TRACKING EVALUATION")
    print("=" * 70)

    print("\n[평가 방법]")
    print("  - CLEAR MOT 메트릭 사용 (motmetrics 라이브러리)")
    print("  - IoU ≥ 0.5 기준으로 GT-Pred 매칭")
    print("  - 프레임별 매칭 후 ID 일관성 평가")
    print()
    print("  주요 지표:")
    print("    - MOTA: Multi-Object Tracking Accuracy (FN, FP, ID Switch 반영)")
    print("    - IDF1: ID F1 Score (ID 일관성 측정)")
    print("    - MOTP: 매칭된 객체의 평균 IoU")
    print("    - ID Switch: 트랙 ID가 바뀐 횟수")

    # Prepare data for tracking evaluation
    track_predictions = {}
    track_gt = {}

    for seq_name in sequences:
        predictions = all_predictions[seq_name]
        gt_data = all_gt_data[seq_name]

        # Convert to tracking format
        track_predictions[seq_name] = {}
        track_gt[seq_name] = {}

        for frame_num, tracks in predictions.items():
            track_predictions[seq_name][frame_num] = [
                (t[0], t[1], t[2], t[3], t[4]) for t in tracks
            ]

        for frame_num, targets in gt_data.items():
            track_gt[seq_name][frame_num] = [
                (t[0], t[1], t[2], t[3], t[4]) for t in targets
            ]

    # Evaluate tracking
    track_metrics = evaluate_tracking_multi_sequence(track_predictions, track_gt, all_ignored)

    print("\n[Tracking 결과]")
    print("-" * 50)
    print(f"  MOTA:        {track_metrics.mota*100:.2f}%")
    print(f"  IDF1:        {track_metrics.idf1*100:.2f}%")
    print(f"  MOTP:        {track_metrics.motp:.4f}")
    print()
    print(f"  ID Switches:      {track_metrics.num_switches:,}")
    print(f"  False Positives:  {track_metrics.num_false_positives:,}")
    print(f"  Misses (FN):      {track_metrics.num_misses:,}")
    print()
    print(f"  Precision:   {track_metrics.precision*100:.2f}%")
    print(f"  Recall:      {track_metrics.recall*100:.2f}%")

    # =========================================================================
    # 3. SPEED EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. SPEED EVALUATION")
    print("=" * 70)

    print("\n[평가 방법]")
    print("  1. Pixel Speed 계산:")
    print("     - 중심점 이동: center = ((x1+x2)/2, (y1+y2)/2)")
    print("     - 속도 = sqrt(dx² + dy²) / frame_gap")
    print()
    print("  2. GT-Prediction 매칭:")
    print("     - 프레임별 IoU ≥ 0.5 기준으로 매칭")
    print("     - TP: 매칭 성공 → 속도 비교 가능")
    print("     - FN: GT 있지만 Pred 없음 → 놓친 차량")
    print("     - FP: Pred 있지만 GT 없음 → 잘못된 검출")
    print()
    print("  3. 속도 정확도 평가 (TP만 대상):")
    print("     - MAE: Mean Absolute Error")
    print("     - RMSE: Root Mean Square Error")
    print("     - Correlation: Pearson 상관계수")
    print()
    print("  4. 속도 분류 평가:")
    print("     - 3클래스: LOW(저속), MEDIUM(중속), HIGH(고속)")
    print("     - Semantic Threshold: LOW < 1.0, HIGH > 5.0 px/frame")

    # Aggregate all predictions and GT for speed evaluation
    all_pred_flat = {}
    all_gt_flat = {}
    all_ignored_list = []

    frame_offset = 0
    for seq_name in sequences:
        predictions = all_predictions[seq_name]
        gt_data = all_gt_data[seq_name]
        ignored = all_ignored[seq_name]

        # Offset frame numbers to avoid collision
        max_frame = max(predictions.keys()) if predictions else 0

        for frame_num, tracks in predictions.items():
            all_pred_flat[frame_offset + frame_num] = tracks

        for frame_num, targets in gt_data.items():
            all_gt_flat[frame_offset + frame_num] = targets

        all_ignored_list.extend(ignored)
        frame_offset += max_frame + 1000  # Add buffer

    # Evaluate speed
    low_th, high_th = THRESHOLD_SEMANTIC
    speed_metrics, comparisons = evaluate_speed_accuracy(
        all_pred_flat, all_gt_flat, all_ignored_list,
        low_threshold=low_th, high_threshold=high_th,
        iou_threshold=0.5
    )

    if speed_metrics:
        print("\n[Speed 결과]")
        print("-" * 50)

        total_gt = speed_metrics.num_tp_samples + speed_metrics.num_fn_samples
        total_pred = speed_metrics.num_tp_samples + speed_metrics.num_fp_samples

        print(f"\n  Coverage Analysis:")
        print(f"    GT 총 샘플:      {total_gt:,}")
        print(f"      - TP (매칭):   {speed_metrics.num_tp_samples:,} ({speed_metrics.gt_coverage*100:.1f}%)")
        print(f"      - FN (놓침):   {speed_metrics.num_fn_samples:,} ({(1-speed_metrics.gt_coverage)*100:.1f}%)")
        print()
        print(f"    Pred 총 샘플:    {total_pred:,}")
        print(f"      - TP (정확):   {speed_metrics.num_tp_samples:,} ({speed_metrics.pred_precision*100:.1f}%)")
        print(f"      - FP (오탐):   {speed_metrics.num_fp_samples:,} ({(1-speed_metrics.pred_precision)*100:.1f}%)")

        print(f"\n  Speed Accuracy (TP only):")
        print(f"    MAE:         {speed_metrics.mae:.3f} px/frame")
        print(f"    RMSE:        {speed_metrics.rmse:.3f} px/frame")
        print(f"    Correlation: {speed_metrics.correlation:.3f}")

        print(f"\n  Speed Classification:")
        print(f"    Class Accuracy: {speed_metrics.class_accuracy*100:.1f}%")
        print(f"      - LOW:    {speed_metrics.low_accuracy*100:.1f}%")
        print(f"      - MEDIUM: {speed_metrics.medium_accuracy*100:.1f}%")
        print(f"      - HIGH:   {speed_metrics.high_accuracy*100:.1f}%")

        # Confusion Matrix
        print_confusion_matrix(speed_metrics.confusion_matrix)

    # =========================================================================
    # 4. LATENCY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. LATENCY ANALYSIS")
    print("=" * 70)

    avg_det_latency = total_det_time / total_frames if total_frames > 0 else 0
    avg_track_latency = total_track_time / total_frames if total_frames > 0 else 0
    total_latency = avg_det_latency + avg_track_latency
    fps = 1000 / total_latency if total_latency > 0 else 0

    print(f"\n  Total Frames:       {total_frames:,}")
    print(f"\n  Detection Latency:  {avg_det_latency:.2f} ms/frame")
    print(f"  Tracking Latency:   {avg_track_latency:.2f} ms/frame")
    print(f"  Total Latency:      {total_latency:.2f} ms/frame")
    print(f"\n  Throughput:         {fps:.1f} FPS")

    # =========================================================================
    # 5. SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. EVALUATION SUMMARY")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    EVALUATION RESULTS                           │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Detection                                                      │")
    print(f"│    AP@0.5:    {det_metrics_50.ap*100:6.2f}%    Precision: {det_metrics_50.precision*100:6.2f}%             │")
    print(f"│    AP@0.75:   {det_metrics_75.ap*100:6.2f}%    Recall:    {det_metrics_50.recall*100:6.2f}%             │")
    print(f"│    COCO mAP:  {coco_map*100:6.2f}%    F1 Score:  {det_metrics_50.f1*100:6.2f}%             │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Tracking                                                       │")
    print(f"│    MOTA:      {track_metrics.mota*100:6.2f}%    ID Switches: {track_metrics.num_switches:6,}            │")
    print(f"│    IDF1:      {track_metrics.idf1*100:6.2f}%    FP:          {track_metrics.num_false_positives:6,}            │")
    print(f"│    MOTP:      {track_metrics.motp:6.4f}     Misses:      {track_metrics.num_misses:6,}            │")
    print("├─────────────────────────────────────────────────────────────────┤")
    if speed_metrics:
        print(f"│  Speed                                                          │")
        print(f"│    GT Coverage:    {speed_metrics.gt_coverage*100:5.1f}%    MAE:  {speed_metrics.mae:5.3f} px/frame         │")
        print(f"│    Pred Precision: {speed_metrics.pred_precision*100:5.1f}%    RMSE: {speed_metrics.rmse:5.3f} px/frame         │")
        print(f"│    Class Accuracy: {speed_metrics.class_accuracy*100:5.1f}%    Corr: {speed_metrics.correlation:5.3f}               │")
        print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Latency                                                        │")
    print(f"│    Detection:  {avg_det_latency:6.2f} ms    Total: {total_latency:6.2f} ms              │")
    print(f"│    Tracking:   {avg_track_latency:6.2f} ms    FPS:   {fps:6.1f}                   │")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Return all metrics
    return {
        'detection': det_results,
        'coco_map': coco_map,
        'tracking': track_metrics,
        'speed': speed_metrics,
        'latency': {
            'detection_ms': avg_det_latency,
            'tracking_ms': avg_track_latency,
            'total_ms': total_latency,
            'fps': fps
        },
        'total_frames': total_frames,
        'num_sequences': len(sequences)
    }


def main():
    parser = argparse.ArgumentParser(description='Full evaluation pipeline')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--max_sequences', type=int, default=None, help='Max sequences')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--conf', type=float, default=None, help='Confidence threshold')
    args = parser.parse_args()

    config = Config() if args.config is None else Config.from_yaml(args.config)

    if args.conf is not None:
        config.detection.confidence_threshold = args.conf

    run_full_evaluation(
        config=config,
        max_sequences=args.max_sequences,
        split=args.split,
        verbose=True
    )


if __name__ == "__main__":
    main()
