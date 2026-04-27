"""Tracking model wrappers for EVA.

Provides unified interface for ByteTrack and DeepSORT trackers.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import cv2


@dataclass
class TrackerArgs:
    """Configuration arguments for ByteTracker.

    Attributes:
        track_thresh: Detection confidence threshold for tracking.
        track_buffer: Number of frames to keep lost tracks.
        match_thresh: IoU matching threshold.
        mot20: MOT20 challenge mode flag.
    """
    track_thresh: float = 0.3
    track_buffer: int = 30
    match_thresh: float = 0.8
    mot20: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> 'TrackerArgs':
        """Create TrackerArgs from a dictionary.

        Args:
            config: Dictionary with tracker configuration.

        Returns:
            TrackerArgs instance.
        """
        return cls(
            track_thresh=config.get('track_thresh', 0.3),
            track_buffer=config.get('track_buffer', 30),
            match_thresh=config.get('match_thresh', 0.8),
            mot20=config.get('mot20', False)
        )


@dataclass
class Track:
    """Single track result."""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.0

    @property
    def box(self) -> Tuple[float, float, float, float]:
        """Get box as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def tlbr(self) -> np.ndarray:
        """Get box as numpy array [x1, y1, x2, y2]."""
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def to_tuple(self, include_score: bool = True) -> Tuple:
        """Convert to tuple format.

        Args:
            include_score: Whether to include score in output.

        Returns:
            Tuple of (track_id, x1, y1, x2, y2) or (track_id, x1, y1, x2, y2, score).
        """
        if include_score:
            return (self.track_id, self.x1, self.y1, self.x2, self.y2, self.score)
        return (self.track_id, self.x1, self.y1, self.x2, self.y2)


class ByteTracker:
    """ByteTrack tracker wrapper."""

    def __init__(self, args: Optional[TrackerArgs] = None):
        """Initialize ByteTracker.

        Args:
            args: TrackerArgs configuration.
        """
        self.args = args or TrackerArgs()
        self.tracker = None

    def _load_tracker(self) -> None:
        """Load the ByteTracker."""
        from ..tracker.byte_tracker import BYTETracker
        self.tracker = BYTETracker(self.args)

    def reset(self) -> None:
        """Reset the tracker state."""
        self._load_tracker()

    def update(
        self,
        detections: List,
        frame_size: Tuple[int, int]
    ) -> List[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of detections. Each detection should be:
                - Detection object with to_bytetrack_format() method, or
                - List/array of [x1, y1, x2, y2, conf]
            frame_size: (height, width) of the frame.

        Returns:
            List of Track objects.
        """
        if self.tracker is None:
            self._load_tracker()

        if len(detections) == 0:
            return []

        # Convert to numpy array
        det_array = []
        for det in detections:
            if hasattr(det, 'to_bytetrack_format'):
                det_array.append(det.to_bytetrack_format())
            else:
                det_array.append(list(det)[:5])  # [x1, y1, x2, y2, conf]

        det_array = np.array(det_array)
        h, w = frame_size

        # Update tracker
        online_targets = self.tracker.update(det_array, [h, w], [h, w])

        # Convert to Track objects
        tracks = []
        for target in online_targets:
            track = Track(
                track_id=target.track_id,
                x1=float(target.tlbr[0]),
                y1=float(target.tlbr[1]),
                x2=float(target.tlbr[2]),
                y2=float(target.tlbr[3]),
                score=float(target.score)
            )
            tracks.append(track)

        return tracks


@dataclass
class DeepSORTArgs:
    """Configuration arguments for DeepSORT tracker."""
    reid_model_path: str = ""
    max_dist: float = 0.2
    max_iou_distance: float = 0.7
    max_age: int = 70
    n_init: int = 3
    nn_budget: int = 100
    use_cuda: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> 'DeepSORTArgs':
        return cls(
            reid_model_path=config.get('reid_model_path', ''),
            max_dist=config.get('max_dist', 0.2),
            max_iou_distance=config.get('max_iou_distance', 0.7),
            max_age=config.get('max_age', 70),
            n_init=config.get('n_init', 3),
            nn_budget=config.get('nn_budget', 100),
            use_cuda=config.get('use_cuda', True)
        )


class DeepSORTTracker:
    """DeepSORT tracker wrapper with Re-ID feature extraction."""

    def __init__(self, args: Optional[DeepSORTArgs] = None):
        """Initialize DeepSORT tracker.

        Args:
            args: DeepSORTArgs configuration.
        """
        self.args = args or DeepSORTArgs()
        self.tracker = None
        self.extractor = None
        self.metric = None

    def _load_tracker(self) -> None:
        """Load DeepSORT tracker and Re-ID model."""
        import torch
        import sys
        from pathlib import Path

        # Add ByteTrack path for imports
        bytetrack_path = Path(__file__).parent.parent.parent / "models" / "tracking" / "ByteTrack"
        if str(bytetrack_path) not in sys.path:
            sys.path.insert(0, str(bytetrack_path))

        from yolox.deepsort_tracker.reid_model import Extractor
        from yolox.deepsort_tracker import kalman_filter
        from yolox.deepsort_tracker.track import Track as DSTrack
        from yolox.deepsort_tracker.detection import Detection as DSDetection

        # Store for later use
        self._kalman_filter = kalman_filter.KalmanFilter()
        self._DSTrack = DSTrack
        self._DSDetection = DSDetection

        # Load Re-ID extractor
        if self.args.reid_model_path:
            self.extractor = Extractor(
                self.args.reid_model_path,
                use_cuda=self.args.use_cuda
            )
        else:
            raise ValueError("reid_model_path is required for DeepSORT")

        # Initialize metric
        self.metric = _NearestNeighborDistanceMetric(
            metric="cosine",
            matching_threshold=self.args.max_dist,
            budget=self.args.nn_budget
        )

        # Initialize tracker
        self.tracker = _DeepSORTCore(
            metric=self.metric,
            max_iou_distance=self.args.max_iou_distance,
            max_age=self.args.max_age,
            n_init=self.args.n_init,
            kalman_filter=self._kalman_filter,
            DSTrack=self._DSTrack
        )

    def reset(self) -> None:
        """Reset the tracker state."""
        if self.extractor is None:
            self._load_tracker()
        else:
            # Just reset tracker state, keep extractor
            self.metric = _NearestNeighborDistanceMetric(
                metric="cosine",
                matching_threshold=self.args.max_dist,
                budget=self.args.nn_budget
            )
            self.tracker = _DeepSORTCore(
                metric=self.metric,
                max_iou_distance=self.args.max_iou_distance,
                max_age=self.args.max_age,
                n_init=self.args.n_init,
                kalman_filter=self._kalman_filter,
                DSTrack=self._DSTrack
            )

    def update(
        self,
        detections: List,
        frame_size: Tuple[int, int],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of Detection objects.
            frame_size: (height, width) of the frame.
            frame: BGR image for Re-ID feature extraction.

        Returns:
            List of Track objects.
        """
        if self.tracker is None:
            self._load_tracker()

        if len(detections) == 0 or frame is None:
            self.tracker.predict()
            self.tracker.increment_ages()
            return self._get_active_tracks()

        # Extract bboxes and confidence
        bboxes = []
        confidences = []
        for det in detections:
            if hasattr(det, 'box'):
                x1, y1, x2, y2 = det.box
                conf = det.confidence
            else:
                x1, y1, x2, y2, conf = det[:5]
            bboxes.append([x1, y1, x2 - x1, y2 - y1])  # tlwh format
            confidences.append(conf)

        bboxes = np.array(bboxes)
        confidences = np.array(confidences)

        # Extract Re-ID features
        features = self._extract_features(bboxes, frame)

        # Create DeepSORT detections
        ds_detections = []
        for i, (bbox, conf, feat) in enumerate(zip(bboxes, confidences, features)):
            ds_detections.append(self._DSDetection(bbox, conf, feat))

        # Update tracker
        self.tracker.predict()
        self.tracker.update(ds_detections)

        return self._get_active_tracks()

    def _extract_features(self, bboxes_tlwh: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Extract Re-ID features from image crops."""
        im_crops = []
        h, w = frame.shape[:2]

        for bbox in bboxes_tlwh:
            x, y, bw, bh = bbox
            x1 = max(int(x), 0)
            y1 = max(int(y), 0)
            x2 = min(int(x + bw), w)
            y2 = min(int(y + bh), h)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                im_crops.append(crop)
            else:
                # Invalid crop, use zeros
                im_crops.append(np.zeros((64, 128, 3), dtype=np.uint8))

        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])

        return features

    def _get_active_tracks(self) -> List[Track]:
        """Get currently active tracks."""
        tracks = []
        for t in self.tracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            bbox = t.to_tlwh()
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            tracks.append(Track(
                track_id=t.track_id,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=1.0  # DeepSORT doesn't maintain score
            ))
        return tracks


class _NearestNeighborDistanceMetric:
    """Nearest neighbor distance metric for Re-ID matching."""

    def __init__(self, metric: str, matching_threshold: float, budget: Optional[int] = None):
        if metric == "cosine":
            self._metric = self._nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    @staticmethod
    def _nn_cosine_distance(x, y):
        x = np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True)
        y = np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)
        distances = 1. - np.dot(x, y.T)
        return distances.min(axis=0)

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


class _DeepSORTCore:
    """Core DeepSORT tracker logic."""

    def __init__(self, metric, max_iou_distance, max_age, n_init, kalman_filter, DSTrack):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = kalman_filter
        self._DSTrack = DSTrack
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):
        # Run matching cascade
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        if features:
            self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            # Gate by threshold
            cost_matrix[cost_matrix > self.metric.matching_threshold] = 1e5
            return cost_matrix

        # Split tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Cascade matching for confirmed tracks
        matches_a, unmatched_tracks_a, unmatched_detections = \
            self._matching_cascade(gated_metric, self.max_age, detections, confirmed_tracks)

        # IOU matching for remaining
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = \
            self._iou_matching(iou_track_candidates, unmatched_detections, detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _matching_cascade(self, metric_fn, max_age, detections, track_indices):
        unmatched_detections = list(range(len(detections)))
        matches = []

        for level in range(max_age):
            if not unmatched_detections:
                break
            track_indices_l = [
                k for k in track_indices
                if self.tracks[k].time_since_update == 1 + level]
            if not track_indices_l:
                continue

            cost_matrix = metric_fn(self.tracks, detections, track_indices_l, unmatched_detections)
            matches_l, _, unmatched_detections = self._linear_assignment(
                cost_matrix, track_indices_l, unmatched_detections)
            matches += matches_l

        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
        return matches, unmatched_tracks, unmatched_detections

    def _iou_matching(self, track_indices, detection_indices, detections):
        if not track_indices or not detection_indices:
            return [], track_indices, detection_indices

        # Compute IOU cost matrix
        track_boxes = np.array([self.tracks[i].to_tlwh() for i in track_indices])
        det_boxes = np.array([detections[i].tlwh for i in detection_indices])

        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        for i, t_box in enumerate(track_boxes):
            for j, d_box in enumerate(det_boxes):
                iou = self._iou(t_box, d_box)
                cost_matrix[i, j] = 1 - iou

        cost_matrix[cost_matrix > self.max_iou_distance] = 1e5
        return self._linear_assignment(cost_matrix, track_indices, detection_indices)

    @staticmethod
    def _iou(tlwh_a, tlwh_b):
        x1 = max(tlwh_a[0], tlwh_b[0])
        y1 = max(tlwh_a[1], tlwh_b[1])
        x2 = min(tlwh_a[0] + tlwh_a[2], tlwh_b[0] + tlwh_b[2])
        y2 = min(tlwh_a[1] + tlwh_a[3], tlwh_b[1] + tlwh_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = tlwh_a[2] * tlwh_a[3]
        area_b = tlwh_b[2] * tlwh_b[3]
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _linear_assignment(self, cost_matrix, track_indices, detection_indices):
        from scipy.optimize import linear_sum_assignment

        if cost_matrix.size == 0:
            return [], track_indices, detection_indices

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_detections = list(detection_indices)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:
                matches.append((track_indices[r], detection_indices[c]))
                unmatched_tracks.remove(track_indices[r])
                unmatched_detections.remove(detection_indices[c])

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(self._DSTrack(
            mean, covariance, self._next_id, 0, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1


def create_tracker(
    tracker_type: str = "bytetrack",
    **kwargs
):
    """Create tracker based on type.

    Args:
        tracker_type: "bytetrack" or "deepsort"
        **kwargs: Tracker-specific arguments

    Returns:
        Tracker instance (ByteTracker or DeepSORTTracker)
    """
    if tracker_type.lower() == "bytetrack":
        args = TrackerArgs(
            track_thresh=kwargs.get('track_thresh', 0.3),
            track_buffer=kwargs.get('track_buffer', 30),
            match_thresh=kwargs.get('match_thresh', 0.8),
            mot20=kwargs.get('mot20', False)
        )
        return ByteTracker(args)

    elif tracker_type.lower() == "deepsort":
        args = DeepSORTArgs(
            reid_model_path=kwargs.get('reid_model_path', ''),
            max_dist=kwargs.get('max_dist', 0.2),
            max_iou_distance=kwargs.get('max_iou_distance', 0.7),
            max_age=kwargs.get('max_age', 70),
            n_init=kwargs.get('n_init', 3),
            nn_budget=kwargs.get('nn_budget', 100),
            use_cuda=kwargs.get('use_cuda', True)
        )
        return DeepSORTTracker(args)

    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
