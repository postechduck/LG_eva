"""Tracking model wrappers for EVA.

Provides unified interface for ByteTrack tracker.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


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
