"""
Detection + Tracking 성능 평가 스크립트 (수정본)
- 버그 수정: 처리한 프레임에 대해서만 평가

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

# Add eva package to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/data/home/pkw_aim25/EVA_2/models/tracking/ByteTrack')

from ultralytics import RTDETR, YOLO
from yolox.tracker.byte_tracker import BYTETracker

# Import from eva package
from eva.utils import compute_iou
from eva.models import TrackerArgs


def parse_ua_detrac_xml(xml_path):
    """UA-DETRAC XML 어노테이션 파싱 (simplified version without ignored regions)"""
    from eva.data import parse_ua_detrac_xml as _parse
    gt_data, _ = _parse(xml_path, include_ignored_regions=False)
    return gt_data


def evaluate_detection(predictions, ground_truths, evaluated_frames, iou_threshold=0.5):
    """Detection 성능 평가 - 처리한 프레임만 평가"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    # 수정: 처리한 프레임에 대해서만 평가
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

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0

    return {
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Avg_IoU': avg_iou
    }


def evaluate_tracking(predictions, ground_truths, evaluated_frames):
    """Tracking 성능 평가 - 처리한 프레임만 평가"""
    acc = mm.MOTAccumulator(auto_id=True)

    for frame_num in sorted(evaluated_frames):
        gt_boxes = ground_truths.get(frame_num, [])
        pred_boxes = predictions.get(frame_num, [])

        gt_ids = [gt[0] for gt in gt_boxes]
        gt_bboxes = [gt[1:5] for gt in gt_boxes]
        pred_ids = [pred[0] for pred in pred_boxes]
        pred_bboxes = [pred[1:5] for pred in pred_boxes]

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

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'mota', 'motp', 'idf1', 'num_switches',
        'num_false_positives', 'num_misses',
        'precision', 'recall'
    ], name='eval')

    return summary


def run_evaluation(sequence_name='MVI_20011', max_frames=None, conf_threshold=0.3):
    """전체 평가 파이프라인 실행"""

    base_path = Path('/data/home/pkw_aim25/EVA_2/data/UA-DETRAC')
    image_dir = base_path / 'DETRAC-Images' / sequence_name
    xml_path = base_path / 'DETRAC-Train-Annotations-XML' / f'{sequence_name}.xml'

    # GT 로드
    print("GT 데이터 로드 중...")
    gt_data = parse_ua_detrac_xml(xml_path)

    # 모델 로드
    print("모델 로드 중...")
    detector = RTDETR('/data/home/pkw_aim25/EVA_2/models/detection/rtdetr-l.pt')
    tracker = BYTETracker(TrackerArgs())

    # 이미지 처리
    image_files = sorted(image_dir.glob('*.jpg'))
    if max_frames:
        image_files = image_files[:max_frames]

    print(f"시퀀스: {sequence_name}")
    print(f"처리할 프레임: {len(image_files)}")
    print(f"Confidence Threshold: {conf_threshold}")
    print("=" * 60)

    # Warm-up
    frame = cv2.imread(str(image_files[0]))
    _ = detector(frame, verbose=False)

    predictions = {}
    evaluated_frames = []
    latencies = []

    print("Detection + Tracking 실행 중...")
    for idx, img_path in enumerate(image_files):
        frame_num = idx + 1
        evaluated_frames.append(frame_num)
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]

        # Detection with timing
        t_start = time.perf_counter()
        results = detector(frame, verbose=False, conf=conf_threshold)
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf])

        # Tracking
        if len(detections) > 0:
            det_array = np.array(detections)
            online_targets = tracker.update(det_array, [h, w], [h, w])
            frame_preds = []
            for track in online_targets:
                track_id = track.track_id
                x1, y1, x2, y2 = track.tlbr
                frame_preds.append((track_id, x1, y1, x2, y2))
            predictions[frame_num] = frame_preds
        else:
            predictions[frame_num] = []

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(image_files)} 프레임 처리 완료")

    print(f"  {len(image_files)}/{len(image_files)} 프레임 처리 완료")

    # 평가 (수정: evaluated_frames만 사용)
    print("\n성능 평가 중...")
    det_metrics = evaluate_detection(predictions, gt_data, evaluated_frames)
    track_metrics = evaluate_tracking(predictions, gt_data, evaluated_frames)
    avg_latency = np.mean(latencies)

    return det_metrics, track_metrics, avg_latency, len(evaluated_frames)


if __name__ == "__main__":
    print("=" * 60)
    print("RT-DETR + ByteTrack 성능 평가 (수정본)")
    print("=" * 60)

    # 전체 시퀀스 평가 (664 프레임)
    det_metrics, track_metrics, latency, num_frames = run_evaluation(
        sequence_name='MVI_20011',
        max_frames=None,  # 전체 프레임
        conf_threshold=0.3
    )

    print("\n" + "=" * 60)
    print("DETECTION 성능")
    print("=" * 60)
    print(f"  처리 프레임:      {num_frames}")
    print(f"  TP:               {det_metrics['TP']}")
    print(f"  FP:               {det_metrics['FP']}")
    print(f"  FN:               {det_metrics['FN']}")
    print(f"  Precision:        {det_metrics['Precision']*100:.2f}%")
    print(f"  Recall:           {det_metrics['Recall']*100:.2f}%")
    print(f"  F1 Score:         {det_metrics['F1']*100:.2f}%")
    print(f"  Average IoU:      {det_metrics['Avg_IoU']*100:.2f}%")

    print("\n" + "=" * 60)
    print("TRACKING 성능")
    print("=" * 60)
    print(track_metrics.to_string())

    print("\n" + "=" * 60)
    print("LATENCY")
    print("=" * 60)
    print(f"  평균 Detection:   {latency:.2f} ms")
    print(f"  처리 가능 FPS:    {1000/latency:.1f}")
    print("=" * 60)
