"""
전체 테스트 데이터셋 평가 스크립트 (v2 - 진행 상황 출력 개선)
- mAP (AP@0.5, AP@0.75, AP@0.5:0.95)
- Detection: Precision, Recall, F1
- Tracking: MOTA, MOTP, IDF1, ID Switches
- Ignored Region 처리 (Center 기준)

Note: This script now uses the eva package for core functionality.
      For new code, consider using: from eva import EvaluationPipeline
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import motmetrics as mm
import time
from collections import defaultdict

# stdout 버퍼링 비활성화
sys.stdout.reconfigure(line_buffering=True)

# Add eva package to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/data/home/pkw_aim25/EVA_2/models/tracking/ByteTrack')

from ultralytics import RTDETR
from yolox.tracker.byte_tracker import BYTETracker

# Import from eva package (backward compatible aliases)
from eva.data import parse_ua_detrac_xml
from eva.utils import compute_iou, is_in_ignored_region as is_in_ignored
from eva.evaluation import compute_ap
from eva.models import TrackerArgs


def evaluate_detection_with_map(all_predictions, all_ground_truths, all_ignored_regions,
                                 iou_thresholds=[0.5]):
    """Detection 성능 평가 + mAP 계산"""

    results = {}

    for iou_thresh in iou_thresholds:
        all_detections = []
        total_gt = 0

        for seq_name in all_predictions.keys():
            predictions = all_predictions[seq_name]
            ground_truths = all_ground_truths[seq_name]
            ignored_regions = all_ignored_regions[seq_name]

            for frame_num in predictions.keys():
                gt_boxes = ground_truths.get(frame_num, [])
                pred_boxes = predictions.get(frame_num, [])

                valid_gt = [gt for gt in gt_boxes if not is_in_ignored(gt[1:5], ignored_regions)]
                total_gt += len(valid_gt)

                gt_matched = [False] * len(valid_gt)

                for pred in pred_boxes:
                    pred_box = pred[1:5]
                    conf = pred[5] if len(pred) > 5 else 1.0

                    if is_in_ignored(pred_box, ignored_regions):
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

        ap = compute_ap(precisions, recalls)

        final_tp = tp_cumsum
        final_fp = fp_cumsum
        final_fn = total_gt - final_tp

        final_precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0
        final_recall = final_tp / total_gt if total_gt > 0 else 0
        final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

        results[iou_thresh] = {
            'AP': ap,
            'Precision': final_precision,
            'Recall': final_recall,
            'F1': final_f1,
            'TP': final_tp,
            'FP': final_fp,
            'FN': final_fn,
            'Total_GT': total_gt
        }

    return results


def evaluate_tracking(all_predictions, all_ground_truths, all_ignored_regions):
    """Tracking 성능 평가 - 시퀀스별 별도 accumulator 사용"""
    accumulators = []
    names = []

    for seq_name in all_predictions.keys():
        acc = mm.MOTAccumulator(auto_id=True)  # 시퀀스별 별도 accumulator
        predictions = all_predictions[seq_name]
        ground_truths = all_ground_truths[seq_name]
        ignored_regions = all_ignored_regions[seq_name]

        for frame_num in sorted(predictions.keys()):
            gt_boxes = ground_truths.get(frame_num, [])
            pred_boxes = predictions.get(frame_num, [])

            valid_gt = [gt for gt in gt_boxes if not is_in_ignored(gt[1:5], ignored_regions)]
            valid_pred = [pred for pred in pred_boxes if not is_in_ignored(pred[1:5], ignored_regions)]

            gt_ids = [gt[0] for gt in valid_gt]
            gt_bboxes = [gt[1:5] for gt in valid_gt]
            pred_ids = [pred[0] for pred in valid_pred]
            pred_bboxes = [pred[1:5] for pred in valid_pred]

            if len(gt_bboxes) > 0 and len(pred_bboxes) > 0:
                distances = np.zeros((len(gt_bboxes), len(pred_bboxes)))
                for i, gt_box in enumerate(gt_bboxes):
                    for j, pred_box in enumerate(pred_bboxes):
                        iou = compute_iou(gt_box, pred_box)
                        distances[i, j] = 1 - iou
                distances[distances > 0.5] = np.nan
            else:
                distances = np.empty((len(gt_bboxes), len(pred_bboxes)))
                distances[:] = np.nan

            acc.update(gt_ids, pred_ids, distances)

        accumulators.append(acc)
        names.append(seq_name)

    mh = mm.metrics.create()
    summary = mh.compute_many(accumulators, names=names, metrics=[
        'mota', 'motp', 'idf1', 'num_switches',
        'num_false_positives', 'num_misses',
        'precision', 'recall'
    ], generate_overall=True)

    return summary


def run_full_evaluation(test_sequences=None, conf_threshold=0.3, max_sequences=None):
    """전체 평가 실행"""

    base_path = Path('/data/home/pkw_aim25/EVA_2/data/UA-DETRAC')
    image_base = base_path / 'DETRAC-Images'
    xml_base = base_path / 'DETRAC-Test-Annotations-XML'

    if test_sequences is None:
        test_sequences = sorted([f.stem for f in xml_base.glob('*.xml')])

    if max_sequences:
        test_sequences = test_sequences[:max_sequences]

    print(f"평가할 시퀀스 수: {len(test_sequences)}", flush=True)
    print(f"Confidence Threshold: {conf_threshold}", flush=True)
    print("=" * 70, flush=True)

    print("모델 로드 중...", flush=True)
    detector = RTDETR('/data/home/pkw_aim25/EVA_2/models/detection/rtdetr-l.pt')

    # Warm-up
    print("Warm-up...", flush=True)
    dummy_img = np.zeros((540, 960, 3), dtype=np.uint8)
    _ = detector(dummy_img, verbose=False)
    print("모델 준비 완료!", flush=True)

    all_predictions = {}
    all_ground_truths = {}
    all_ignored_regions = {}
    all_latencies = []
    total_frames = 0

    start_time = time.time()

    for seq_idx, seq_name in enumerate(test_sequences):
        seq_start = time.time()
        print(f"\n[{seq_idx+1}/{len(test_sequences)}] {seq_name} 처리 중...", flush=True)

        image_dir = image_base / seq_name
        xml_path = xml_base / f'{seq_name}.xml'

        if not image_dir.exists():
            print(f"  이미지 폴더 없음: {image_dir}", flush=True)
            continue

        gt_data, ignored_regions = parse_ua_detrac_xml(xml_path)
        all_ground_truths[seq_name] = gt_data
        all_ignored_regions[seq_name] = ignored_regions

        tracker = BYTETracker(TrackerArgs())

        image_files = sorted(image_dir.glob('*.jpg'))
        predictions = {}

        for idx, img_path in enumerate(image_files):
            frame_num = idx + 1
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]

            t_start = time.perf_counter()
            results = detector(frame, verbose=False, conf=conf_threshold)
            t_end = time.perf_counter()
            all_latencies.append((t_end - t_start) * 1000)

            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append([x1, y1, x2, y2, conf])

            if len(detections) > 0:
                det_array = np.array(detections)
                online_targets = tracker.update(det_array, [h, w], [h, w])
                frame_preds = []
                for track in online_targets:
                    track_id = track.track_id
                    x1, y1, x2, y2 = track.tlbr
                    conf = track.score
                    frame_preds.append((track_id, x1, y1, x2, y2, conf))
                predictions[frame_num] = frame_preds
            else:
                predictions[frame_num] = []

            total_frames += 1

        all_predictions[seq_name] = predictions
        seq_time = time.time() - seq_start
        print(f"  {len(image_files)} 프레임 처리 ({seq_time:.1f}s)", flush=True)

    total_time = time.time() - start_time
    print(f"\n총 처리 시간: {total_time:.1f}s", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("성능 평가 중...", flush=True)

    # Detection + mAP 평가
    iou_thresholds = [0.5, 0.75]
    det_results = evaluate_detection_with_map(
        all_predictions, all_ground_truths, all_ignored_regions,
        iou_thresholds=iou_thresholds
    )

    # COCO-style mAP
    print("COCO mAP 계산 중...", flush=True)
    coco_thresholds = np.arange(0.5, 1.0, 0.05)
    coco_results = evaluate_detection_with_map(
        all_predictions, all_ground_truths, all_ignored_regions,
        iou_thresholds=coco_thresholds
    )
    coco_map = np.mean([coco_results[t]['AP'] for t in coco_thresholds])

    # Tracking 평가
    print("Tracking 메트릭 계산 중...", flush=True)
    track_metrics = evaluate_tracking(all_predictions, all_ground_truths, all_ignored_regions)

    avg_latency = np.mean(all_latencies) if all_latencies else 0

    return det_results, coco_map, track_metrics, avg_latency, total_frames, len(test_sequences)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sequences', type=int, default=None, help='Max sequences to evaluate')
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("UA-DETRAC 테스트셋 평가", flush=True)
    print("RT-DETR + ByteTrack", flush=True)
    print("=" * 70, flush=True)

    det_results, coco_map, track_metrics, latency, total_frames, num_sequences = run_full_evaluation(
        conf_threshold=0.3,
        max_sequences=args.max_sequences
    )

    print("\n" + "=" * 70, flush=True)
    print("DETECTION 성능", flush=True)
    print("=" * 70, flush=True)
    print(f"  총 시퀀스:        {num_sequences}", flush=True)
    print(f"  총 프레임:        {total_frames}", flush=True)
    print(flush=True)

    for iou_thresh, metrics in det_results.items():
        print(f"  [IoU Threshold = {iou_thresh}]", flush=True)
        print(f"    AP:             {metrics['AP']*100:.2f}%", flush=True)
        print(f"    Precision:      {metrics['Precision']*100:.2f}%", flush=True)
        print(f"    Recall:         {metrics['Recall']*100:.2f}%", flush=True)
        print(f"    F1 Score:       {metrics['F1']*100:.2f}%", flush=True)
        print(f"    TP/FP/FN:       {metrics['TP']}/{metrics['FP']}/{metrics['FN']}", flush=True)
        print(flush=True)

    print(f"  [COCO mAP (0.5:0.95)]", flush=True)
    print(f"    mAP:            {coco_map*100:.2f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("TRACKING 성능", flush=True)
    print("=" * 70, flush=True)
    print(track_metrics.to_string(), flush=True)

    print("\n" + "=" * 70, flush=True)
    print("LATENCY", flush=True)
    print("=" * 70, flush=True)
    print(f"  평균 Detection:   {latency:.2f} ms", flush=True)
    print(f"  처리 가능 FPS:    {1000/latency:.1f}" if latency > 0 else "  처리 가능 FPS:    N/A", flush=True)
    print("=" * 70, flush=True)
