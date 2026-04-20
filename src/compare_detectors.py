"""
Detection 모델 비교: RT-DETR vs YOLOv8x
- Tracking 없이 Detection만 평가

Note: This script now uses the eva package for core functionality.
      For new code, consider using: scripts/compare.py
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import time

# Add eva package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

sys.stdout.reconfigure(line_buffering=True)

from ultralytics import RTDETR, YOLO

# Import from eva package
from eva.data import parse_ua_detrac_xml as _parse_xml
from eva.utils import compute_iou, is_in_ignored_region as is_in_ignored
from eva.evaluation import compute_ap


def parse_ua_detrac_xml(xml_path):
    """UA-DETRAC XML 파싱 (wrapper for backward compatibility)"""
    gt_data, ignored_regions = _parse_xml(xml_path)
    # Convert to format without target_id (original format)
    gt_data_simple = {}
    for frame_num, targets in gt_data.items():
        gt_data_simple[frame_num] = [t[1:5] for t in targets]  # Remove target_id
    return gt_data_simple, ignored_regions


def evaluate_detector(detector, detector_name, test_sequences, base_path, conf_threshold=0.3):
    """단일 detector 평가"""

    image_base = base_path / 'DETRAC-Images'
    xml_base = base_path / 'DETRAC-Test-Annotations-XML'

    all_detections = []  # (conf, is_tp)
    total_gt = 0
    all_latencies = []
    total_frames = 0

    for seq_idx, seq_name in enumerate(test_sequences):
        print(f"  [{seq_idx+1}/{len(test_sequences)}] {seq_name}", end="", flush=True)

        image_dir = image_base / seq_name
        xml_path = xml_base / f'{seq_name}.xml'

        if not image_dir.exists():
            print(" - 스킵")
            continue

        gt_data, ignored_regions = parse_ua_detrac_xml(xml_path)
        image_files = sorted(image_dir.glob('*.jpg'))

        seq_start = time.time()

        for idx, img_path in enumerate(image_files):
            frame_num = idx + 1
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            # Detection
            t_start = time.perf_counter()
            results = detector(frame, verbose=False, conf=conf_threshold)
            t_end = time.perf_counter()
            all_latencies.append((t_end - t_start) * 1000)

            # GT 처리
            gt_boxes = gt_data.get(frame_num, [])
            valid_gt = [gt for gt in gt_boxes if not is_in_ignored(gt, ignored_regions)]
            total_gt += len(valid_gt)

            gt_matched = [False] * len(valid_gt)

            # Detection 결과 처리
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        pred_box = (x1, y1, x2, y2)

                        if is_in_ignored(pred_box, ignored_regions):
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
                            all_detections.append((conf, True))
                        else:
                            all_detections.append((conf, False))

            total_frames += 1

        seq_time = time.time() - seq_start
        print(f" ({len(image_files)} frames, {seq_time:.1f}s)", flush=True)

    # AP 계산
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
        'FPS': 1000 / avg_latency,
        'Total_Frames': total_frames
    }


def main():
    print("=" * 70)
    print("Detection 모델 비교: RT-DETR vs YOLOv8x")
    print("=" * 70)

    base_path = Path('/data/home/pkw_aim25/EVA_2/data/UA-DETRAC')
    xml_base = base_path / 'DETRAC-Test-Annotations-XML'

    # 테스트 시퀀스 (전체 40개)
    test_sequences = sorted([f.stem for f in xml_base.glob('*.xml')])

    print(f"\n데이터셋: UA-DETRAC Test Set")
    print(f"시퀀스 수: {len(test_sequences)}")
    print(f"Confidence Threshold: 0.3")
    print(f"IoU Threshold: 0.5")
    print("=" * 70)

    results = {}

    # RT-DETR 평가
    print("\n[1] RT-DETR-L 평가 중...")
    rtdetr = RTDETR('/data/home/pkw_aim25/EVA_2/models/detection/rtdetr-l.pt')
    # Warm-up
    dummy = np.zeros((540, 960, 3), dtype=np.uint8)
    _ = rtdetr(dummy, verbose=False)
    results['RT-DETR-L'] = evaluate_detector(rtdetr, 'RT-DETR-L', test_sequences, base_path)
    del rtdetr

    # YOLOv8x 평가
    print("\n[2] YOLOv8x 평가 중...")
    yolo = YOLO('/data/home/pkw_aim25/EVA_2/models/detection/yolov8x.pt')
    # Warm-up
    _ = yolo(dummy, verbose=False)
    results['YOLOv8x'] = evaluate_detector(yolo, 'YOLOv8x', test_sequences, base_path)
    del yolo

    # 결과 출력
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)

    print("\n### Detection 성능 (IoU=0.5)")
    print("-" * 70)
    print(f"{'모델':<15} {'AP':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['AP']*100:>6.2f}%   {metrics['Precision']*100:>8.2f}%   {metrics['Recall']*100:>6.2f}%   {metrics['F1']*100:>6.2f}%")

    print("\n### TP / FP / FN")
    print("-" * 70)
    print(f"{'모델':<15} {'TP':<12} {'FP':<12} {'FN':<12} {'Total GT':<12}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['TP']:<12} {metrics['FP']:<12} {metrics['FN']:<12} {metrics['Total_GT']:<12}")

    print("\n### Latency")
    print("-" * 70)
    print(f"{'모델':<15} {'Latency (ms)':<15} {'FPS':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['Latency_ms']:>10.2f}     {metrics['FPS']:>6.1f}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    main()
