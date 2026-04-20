"""
Detection 결과 시각화 영상 생성
- RT-DETR detection 결과만 표시 (tracking 없음)

Note: This script now uses the eva package for core functionality.
      For new code, consider using: scripts/visualize.py
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import time

# Add eva package to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/data/home/pkw_aim25/EVA_2/models/tracking/ByteTrack')

from ultralytics import RTDETR

# Can optionally use eva visualization utilities
# from eva.utils.visualization import draw_detection_box, draw_info_overlay


def create_detection_video(sequence_name='MVI_39031', max_frames=300, conf_threshold=0.3):
    """Detection 결과 시각화 영상 생성"""

    base_path = Path('/data/home/pkw_aim25/EVA_2/data/UA-DETRAC')
    image_dir = base_path / 'DETRAC-Images' / sequence_name
    output_path = Path('/data/home/pkw_aim25/EVA_2/outputs') / f'detection_only_{sequence_name}.mp4'

    # 이미지 파일 목록
    image_files = sorted(image_dir.glob('*.jpg'))
    if max_frames:
        image_files = image_files[:max_frames]

    print(f"시퀀스: {sequence_name}")
    print(f"프레임 수: {len(image_files)}")
    print(f"Confidence Threshold: {conf_threshold}")

    # 모델 로드
    print("모델 로드 중...")
    detector = RTDETR('/data/home/pkw_aim25/EVA_2/models/detection/rtdetr-l.pt')

    # Warm-up
    frame = cv2.imread(str(image_files[0]))
    _ = detector(frame, verbose=False)

    # 비디오 설정
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 25.0, (w, h))

    # COCO 클래스 이름
    class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    colors = {
        2: (0, 255, 0),    # car - green
        3: (255, 0, 0),    # motorcycle - blue
        5: (0, 165, 255),  # bus - orange
        7: (0, 0, 255)     # truck - red
    }

    print("Detection 및 영상 생성 중...")
    latencies = []

    for idx, img_path in enumerate(image_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Detection with timing
        t_start = time.perf_counter()
        results = detector(frame, verbose=False, conf=conf_threshold)
        t_end = time.perf_counter()
        latency = (t_end - t_start) * 1000
        latencies.append(latency)

        # Detection 결과 그리기
        det_count = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in class_names:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])

                    color = colors.get(cls, (255, 255, 255))
                    label = f"{class_names[cls]} {conf:.2f}"

                    # 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 라벨 배경
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)

                    # 라벨 텍스트
                    cv2.putText(frame, label, (x1, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    det_count += 1

        # 프레임 정보 표시
        info_text = f"Frame: {idx+1}/{len(image_files)} | Detections: {det_count} | Latency: {latency:.1f}ms"
        cv2.rectangle(frame, (0, 0), (500, 30), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 모델 정보 표시
        model_text = f"RT-DETR-L | conf={conf_threshold}"
        cv2.rectangle(frame, (w-220, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, model_text, (w-210, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        out.write(frame)

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(image_files)} 프레임 처리 완료")

    out.release()

    avg_latency = np.mean(latencies)
    print(f"\n완료!")
    print(f"출력 파일: {output_path}")
    print(f"평균 Latency: {avg_latency:.2f} ms")
    print(f"처리 FPS: {1000/avg_latency:.1f}")

    return str(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, default='MVI_39031', help='Sequence name')
    parser.add_argument('--max_frames', type=int, default=300, help='Max frames')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    args = parser.parse_args()

    create_detection_video(
        sequence_name=args.sequence,
        max_frames=args.max_frames,
        conf_threshold=args.conf
    )
