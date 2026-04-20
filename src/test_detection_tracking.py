"""
Detection + Tracking 테스트 스크립트
- RT-DETR로 차량 검출
- ByteTrack으로 객체 추적

Note: This script now uses the eva package for core functionality.
      For new code, consider using: from eva import DetectionTrackingPipeline
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add eva package and ByteTrack to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/data/home/pkw_aim25/EVA_2/models/tracking/ByteTrack')

from ultralytics import RTDETR
from yolox.tracker.byte_tracker import BYTETracker

# Import TrackerArgs from eva package
from eva.models import TrackerArgs

# Override track_thresh for this demo (original was 0.5)
TrackerArgs.track_thresh = 0.5

def run_detection_tracking(
    image_dir: str,
    output_path: str,
    model_path: str = '/data/home/pkw_aim25/EVA_2/models/detection/rtdetr-l.pt',
    max_frames: int = 100,  # 테스트용으로 100프레임만
    conf_thresh: float = 0.5
):
    """Detection + Tracking 파이프라인 실행"""

    # 모델 로드
    print(f"Loading RT-DETR model from {model_path}")
    detector = RTDETR(model_path)

    # ByteTracker 초기화
    tracker = BYTETracker(TrackerArgs())

    # 이미지 파일 목록
    image_files = sorted(Path(image_dir).glob('*.jpg'))[:max_frames]
    print(f"Processing {len(image_files)} frames")

    # 첫 이미지로 비디오 설정
    first_img = cv2.imread(str(image_files[0]))
    h, w = first_img.shape[:2]

    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

    # 색상 팔레트 (트랙 ID별 색상)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

    # 결과 저장용
    results_log = []

    for frame_idx, img_path in enumerate(tqdm(image_files, desc="Processing")):
        frame = cv2.imread(str(img_path))

        # RT-DETR Detection
        # 차량 클래스: car(2), motorcycle(3), bus(5), truck(7) in COCO
        results = detector(frame, verbose=False, conf=conf_thresh)

        # Detection 결과 추출
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                # 차량 관련 클래스만 필터링 (COCO: car=2, motorcycle=3, bus=5, truck=7)
                if cls in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf])

        # ByteTrack 업데이트
        if len(detections) > 0:
            detections = np.array(detections)
            online_targets = tracker.update(
                detections,
                [h, w],
                [h, w]
            )
        else:
            online_targets = []

        # 시각화
        for track in online_targets:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr.astype(int)

            # 트랙 ID별 색상
            color = colors[track_id % 1000].tolist()

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ID 라벨
            label = f"ID:{track_id}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 로그 저장
            results_log.append({
                'frame': frame_idx + 1,
                'track_id': track_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

        # 프레임 정보
        cv2.putText(frame, f"Frame: {frame_idx + 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(online_targets)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

    out.release()
    print(f"\nOutput saved to: {output_path}")
    print(f"Total tracks logged: {len(results_log)}")

    return results_log

if __name__ == "__main__":
    # 테스트 실행
    IMAGE_DIR = "/data/home/pkw_aim25/EVA_2/data/UA-DETRAC/DETRAC-Images/MVI_20011"
    OUTPUT_PATH = "/data/home/pkw_aim25/EVA_2/outputs/test_tracking_MVI_20011.mp4"

    results = run_detection_tracking(
        image_dir=IMAGE_DIR,
        output_path=OUTPUT_PATH,
        max_frames=100  # 테스트용 100프레임
    )

    print("\n=== 테스트 완료 ===")
    print(f"처리 프레임: 100")
    print(f"출력 파일: {OUTPUT_PATH}")
