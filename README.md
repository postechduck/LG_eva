# EVA_2: Vehicle Detection & Tracking Pipeline

UA-DETRAC 데이터셋을 이용한 차량 검출 및 추적 파이프라인

---

## 프로젝트 구조

```
EVA_2/
├── configs/                    # 설정 파일 (예정)
├── data/                       # 데이터셋
│   ├── UA-DETRAC/             # UA-DETRAC 데이터셋
│   │   ├── DETRAC-Images/     # 이미지 시퀀스 (100개 시퀀스)
│   │   ├── DETRAC-Train-Annotations-XML/  # 학습 어노테이션 (60개)
│   │   ├── DETRAC-Test-Annotations-XML/   # 테스트 어노테이션 (40개)
│   │   └── ...
│   └── BrnoCompSpeed/         # BrnoCompSpeed 데이터셋 (속도 추정용)
├── models/                     # 모델
│   ├── detection/             # 검출 모델
│   │   ├── rtdetr-l.pt        # RT-DETR-L (메인 검출기)
│   │   ├── yolov8x.pt         # YOLOv8x (비교용)
│   │   └── RT-DETR/           # RT-DETR 소스코드
│   └── tracking/              # 추적 모델
│       └── ByteTrack/         # ByteTrack 소스코드
│           └── yolox/tracker/byte_tracker.py  # ByteTracker 핵심 코드
├── outputs/                    # 출력 결과
│   ├── baseline_evaluation_results.md  # 베이스라인 평가 결과
│   ├── performance_analysis.md         # 성능 분석 보고서
│   ├── *.mp4                           # 시각화 영상
│   └── *.jpg                           # 샘플 이미지
├── src/                        # 소스 코드
│   ├── evaluate_full_test.py           # 전체 테스트셋 평가 (메인)
│   ├── evaluate_performance_fixed.py   # 단일 시퀀스 평가
│   ├── test_detection_tracking.py      # Detection+Tracking 테스트
│   ├── visualize_detection.py          # Detection 시각화 영상 생성
│   └── compare_detectors.py            # Detection 모델 비교
└── ppt/                        # 발표 자료
```

---

## 소스 코드 설명

### 1. evaluate_full_test.py (메인 평가 스크립트)

**용도**: UA-DETRAC 테스트셋 전체(40개 시퀀스) 평가

**주요 기능**:
- RT-DETR + ByteTrack 파이프라인 실행
- Detection 평가: AP, Precision, Recall, F1, COCO mAP
- Tracking 평가: MOTA, MOTP, IDF1, ID Switches
- Ignored Region 처리 (Center 기준)

**핵심 함수**:
```python
# UA-DETRAC XML 어노테이션 파싱
parse_ua_detrac_xml(xml_path) -> (gt_data, ignored_regions)

# Detection 성능 평가 (mAP 포함)
evaluate_detection_with_map(predictions, ground_truths, ignored_regions, iou_thresholds)

# Tracking 성능 평가 (시퀀스별 별도 accumulator)
evaluate_tracking(all_predictions, all_ground_truths, all_ignored_regions)

# 전체 평가 실행
run_full_evaluation(test_sequences, conf_threshold, max_sequences)
```

**실행 방법**:
```bash
python src/evaluate_full_test.py
python src/evaluate_full_test.py --max_sequences 5  # 5개 시퀀스만 평가
```

---

### 2. evaluate_performance_fixed.py (단일 시퀀스 평가)

**용도**: 단일 시퀀스에 대한 빠른 평가/디버깅

**특징**:
- 특정 시퀀스만 평가
- 프레임 수 제한 가능
- 개발/디버깅용

**실행 방법**:
```bash
python src/evaluate_performance_fixed.py
```

---

### 3. test_detection_tracking.py (파이프라인 테스트)

**용도**: Detection + Tracking 파이프라인 동작 테스트 및 시각화

**기능**:
- RT-DETR로 차량 검출
- ByteTrack으로 객체 추적
- 추적 결과 시각화 영상 생성

**출력**: `outputs/test_tracking_*.mp4`

**실행 방법**:
```bash
python src/test_detection_tracking.py
```

---

### 4. visualize_detection.py (Detection 시각화)

**용도**: Detection 결과만 시각화 (Tracking 없음)

**기능**:
- 클래스별 색상 구분 (car, motorcycle, bus, truck)
- Confidence score 표시
- Latency 실시간 표시

**실행 방법**:
```bash
python src/visualize_detection.py --sequence MVI_39031 --max_frames 300 --conf 0.3
```

---

### 5. compare_detectors.py (Detection 모델 비교)

**용도**: RT-DETR vs YOLOv8x 성능 비교

**비교 항목**:
- AP, Precision, Recall, F1
- TP, FP, FN
- Latency, FPS

**실행 방법**:
```bash
python src/compare_detectors.py
```

---

## 모델 설정

### Detection: RT-DETR-L

```python
from ultralytics import RTDETR

detector = RTDETR('models/detection/rtdetr-l.pt')
results = detector(frame, conf=0.3, verbose=False)

# COCO 클래스 필터링 (차량만)
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
```

### Tracking: ByteTrack

```python
from yolox.tracker.byte_tracker import BYTETracker

class TrackerArgs:
    track_thresh = 0.3   # 고신뢰 detection threshold
    track_buffer = 30    # lost track 유지 프레임 수
    match_thresh = 0.8   # IoU 매칭 threshold
    mot20 = False

tracker = BYTETracker(TrackerArgs())
online_targets = tracker.update(detections, [h, w], [h, w])
```

---

## 데이터셋

### UA-DETRAC

- **용도**: 차량 검출 및 추적 평가
- **구성**:
  - Train: 60개 시퀀스
  - Test: 40개 시퀀스, 56,340 프레임
- **어노테이션**: XML 형식 (bbox, track_id, ignored_region)
- **해상도**: 960x540

### BrnoCompSpeed

- **용도**: 속도 추정 평가 (예정)
- **특징**: 실제 차량 속도 GT 포함

---

## 평가 지표

### Detection
| 지표 | 설명 |
|------|------|
| AP@0.5 | IoU 0.5 기준 Average Precision |
| AP@0.75 | IoU 0.75 기준 Average Precision |
| COCO mAP | IoU 0.5:0.95 평균 AP |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 * Precision * Recall / (Precision + Recall) |

### Tracking
| 지표 | 설명 |
|------|------|
| MOTA | 1 - (FN + FP + IDSW) / GT |
| MOTP | 매칭된 bbox의 평균 (1 - IoU) |
| IDF1 | ID 일관성 측정 (높을수록 좋음) |
| ID Switches | 트랙 ID 변경 횟수 |

---

## 현재 성능 (2026-04-13)

```
Detection (IoU=0.5):
  AP:        77.43%
  Precision: 74.83%
  Recall:    86.77%
  F1:        80.36%

Tracking:
  MOTA:      57.42%
  MOTP:      0.151 (IoU ≈ 0.85)
  IDF1:      75.45%
  ID Switches: 1,068

Latency:
  Detection: 21.77 ms
  FPS:       45.9
```

---

## 의존성

```
ultralytics    # RT-DETR, YOLOv8
opencv-python  # 이미지/영상 처리
numpy          # 수치 연산
motmetrics     # MOT 평가 (MOTA, IDF1 등)
tqdm           # 진행 표시
```

---

## 참고 사항

### IDF1 측정 버그 수정 (2026-04-13)

**이전 (잘못된 방식)**:
```python
acc = mm.MOTAccumulator(auto_id=True)
for seq_name in all_predictions.keys():
    # 모든 시퀀스를 하나의 accumulator에 넣음 → 잘못됨
    acc.update(gt_ids, pred_ids, distances)
```

**수정 (올바른 방식)**:
```python
accumulators = []
for seq_name in all_predictions.keys():
    acc = mm.MOTAccumulator(auto_id=True)  # 시퀀스별 별도 생성
    # ...
    accumulators.append(acc)

summary = mh.compute_many(accumulators, names=names, generate_overall=True)
```

- 이전 IDF1: 13.13% (잘못됨)
- 수정 후 IDF1: 75.45% (정확함)
