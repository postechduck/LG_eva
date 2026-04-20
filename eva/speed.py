"""Pixel speed calculation and classification for EVA.

Calculates speed from tracking results and classifies into low/medium/high.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np


class SpeedClass(Enum):
    """Speed classification."""
    LOW = "저속"
    MEDIUM = "중속"
    HIGH = "고속"


@dataclass
class SpeedResult:
    """Speed calculation result for a single track at a frame."""
    track_id: int
    frame_num: int
    center_x: float
    center_y: float
    dx: float = 0.0
    dy: float = 0.0
    pixel_speed: float = 0.0  # pixels/frame
    speed_class: Optional[SpeedClass] = None

    @property
    def pixel_speed_per_sec(self) -> float:
        """Pixel speed per second (assuming 25 fps)."""
        return self.pixel_speed * 25.0


@dataclass
class TrackSpeedStats:
    """Speed statistics for a single track."""
    track_id: int
    num_frames: int
    avg_speed: float
    max_speed: float
    min_speed: float
    std_speed: float
    speed_class: SpeedClass


def calculate_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Calculate center point of a bounding box.

    Args:
        box: (x1, y1, x2, y2) bounding box.

    Returns:
        (center_x, center_y) tuple.
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def calculate_pixel_speed(
    predictions: Dict[int, List[Tuple]],
    fps: float = 25.0
) -> Dict[int, List[SpeedResult]]:
    """Calculate pixel speed for all tracks.

    Args:
        predictions: Dict mapping frame_num -> [(track_id, x1, y1, x2, y2, ...), ...].
        fps: Frames per second of the video.

    Returns:
        Dict mapping track_id -> list of SpeedResult.
    """
    # Reorganize by track_id
    track_data: Dict[int, List[Tuple[int, float, float]]] = {}  # track_id -> [(frame, cx, cy), ...]

    for frame_num, tracks in predictions.items():
        for track in tracks:
            track_id = track[0]
            box = track[1:5]
            cx, cy = calculate_center(box)

            if track_id not in track_data:
                track_data[track_id] = []
            track_data[track_id].append((frame_num, cx, cy))

    # Sort by frame number and calculate speeds
    results: Dict[int, List[SpeedResult]] = {}

    for track_id, frames in track_data.items():
        frames.sort(key=lambda x: x[0])  # Sort by frame number

        results[track_id] = []

        for i, (frame_num, cx, cy) in enumerate(frames):
            if i == 0:
                # First frame: no previous frame to compare
                result = SpeedResult(
                    track_id=track_id,
                    frame_num=frame_num,
                    center_x=cx,
                    center_y=cy,
                    dx=0.0,
                    dy=0.0,
                    pixel_speed=0.0
                )
            else:
                prev_frame, prev_cx, prev_cy = frames[i - 1]

                # Check if consecutive frame
                if frame_num - prev_frame == 1:
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    pixel_speed = math.sqrt(dx * dx + dy * dy)
                else:
                    # Non-consecutive: interpolate or mark as gap
                    frame_gap = frame_num - prev_frame
                    dx = (cx - prev_cx) / frame_gap
                    dy = (cy - prev_cy) / frame_gap
                    pixel_speed = math.sqrt(dx * dx + dy * dy)

                result = SpeedResult(
                    track_id=track_id,
                    frame_num=frame_num,
                    center_x=cx,
                    center_y=cy,
                    dx=dx,
                    dy=dy,
                    pixel_speed=pixel_speed
                )

            results[track_id].append(result)

    return results


def classify_speeds(
    speed_results: Dict[int, List[SpeedResult]],
    low_threshold: float = 5.0,
    high_threshold: float = 20.0
) -> Dict[int, List[SpeedResult]]:
    """Classify speeds into low/medium/high categories.

    Args:
        speed_results: Dict from calculate_pixel_speed.
        low_threshold: Speed below this is LOW (pixels/frame).
        high_threshold: Speed above this is HIGH (pixels/frame).

    Returns:
        Same dict with speed_class filled in.
    """
    for track_id, results in speed_results.items():
        for result in results:
            if result.pixel_speed < low_threshold:
                result.speed_class = SpeedClass.LOW
            elif result.pixel_speed > high_threshold:
                result.speed_class = SpeedClass.HIGH
            else:
                result.speed_class = SpeedClass.MEDIUM

    return speed_results


def compute_track_statistics(
    speed_results: Dict[int, List[SpeedResult]],
    low_threshold: float = 5.0,
    high_threshold: float = 20.0
) -> Dict[int, TrackSpeedStats]:
    """Compute speed statistics for each track.

    Args:
        speed_results: Dict from calculate_pixel_speed.
        low_threshold: Threshold for LOW speed.
        high_threshold: Threshold for HIGH speed.

    Returns:
        Dict mapping track_id -> TrackSpeedStats.
    """
    stats: Dict[int, TrackSpeedStats] = {}

    for track_id, results in speed_results.items():
        # Skip first frame (no speed) and get speeds
        speeds = [r.pixel_speed for r in results[1:] if r.pixel_speed > 0]

        if len(speeds) == 0:
            continue

        avg_speed = np.mean(speeds)

        # Classify based on average speed
        if avg_speed < low_threshold:
            speed_class = SpeedClass.LOW
        elif avg_speed > high_threshold:
            speed_class = SpeedClass.HIGH
        else:
            speed_class = SpeedClass.MEDIUM

        stats[track_id] = TrackSpeedStats(
            track_id=track_id,
            num_frames=len(results),
            avg_speed=avg_speed,
            max_speed=max(speeds),
            min_speed=min(speeds),
            std_speed=np.std(speeds),
            speed_class=speed_class
        )

    return stats


def analyze_speed_distribution(
    speed_results: Dict[int, List[SpeedResult]]
) -> Dict:
    """Analyze overall speed distribution.

    Args:
        speed_results: Dict from calculate_pixel_speed.

    Returns:
        Dict with distribution statistics.
    """
    all_speeds = []

    for track_id, results in speed_results.items():
        for r in results[1:]:  # Skip first frame
            if r.pixel_speed > 0:
                all_speeds.append(r.pixel_speed)

    if len(all_speeds) == 0:
        return {}

    all_speeds = np.array(all_speeds)

    return {
        'count': len(all_speeds),
        'mean': float(np.mean(all_speeds)),
        'std': float(np.std(all_speeds)),
        'min': float(np.min(all_speeds)),
        'max': float(np.max(all_speeds)),
        'median': float(np.median(all_speeds)),
        'percentile_25': float(np.percentile(all_speeds, 25)),
        'percentile_75': float(np.percentile(all_speeds, 75)),
        'percentile_90': float(np.percentile(all_speeds, 90)),
        'percentile_95': float(np.percentile(all_speeds, 95)),
    }


def get_suggested_thresholds(
    speed_results: Dict[int, List[SpeedResult]]
) -> Tuple[float, float]:
    """Suggest thresholds based on data distribution.

    Uses 33rd and 67th percentiles as default thresholds.

    Args:
        speed_results: Dict from calculate_pixel_speed.

    Returns:
        (low_threshold, high_threshold) tuple.
    """
    all_speeds = []

    for track_id, results in speed_results.items():
        for r in results[1:]:
            if r.pixel_speed > 0:
                all_speeds.append(r.pixel_speed)

    if len(all_speeds) == 0:
        return 5.0, 20.0  # Default values

    low_threshold = float(np.percentile(all_speeds, 33))
    high_threshold = float(np.percentile(all_speeds, 67))

    return low_threshold, high_threshold


# Predefined threshold options
THRESHOLD_UNIFORM = (0.5, 1.4)  # 균등 분할 (33/33/33%)
THRESHOLD_SEMANTIC = (1.0, 5.0)  # 의미 기반


def classify_speed_value(
    pixel_speed: float,
    low_threshold: float = 1.0,
    high_threshold: float = 5.0
) -> SpeedClass:
    """Classify a single speed value.

    Args:
        pixel_speed: Speed in pixels/frame.
        low_threshold: Below this is LOW.
        high_threshold: Above this is HIGH.

    Returns:
        SpeedClass enum value.
    """
    if pixel_speed < low_threshold:
        return SpeedClass.LOW
    elif pixel_speed > high_threshold:
        return SpeedClass.HIGH
    else:
        return SpeedClass.MEDIUM


def get_track_trajectory(
    predictions: Dict[int, List[Tuple]],
    track_id: int
) -> List[Tuple[int, float, float, float]]:
    """Get trajectory (frame, x, y, speed) for a specific track.

    Args:
        predictions: Dict mapping frame_num -> [(track_id, x1, y1, x2, y2, ...), ...].
        track_id: Target track ID.

    Returns:
        List of (frame_num, center_x, center_y, pixel_speed) tuples.
    """
    # Collect all frames for this track
    track_frames = []

    for frame_num, tracks in predictions.items():
        for track in tracks:
            if track[0] == track_id:
                box = track[1:5]
                cx, cy = calculate_center(box)
                track_frames.append((frame_num, cx, cy))
                break

    # Sort by frame number
    track_frames.sort(key=lambda x: x[0])

    # Calculate speeds
    trajectory = []
    for i, (frame_num, cx, cy) in enumerate(track_frames):
        if i == 0:
            speed = 0.0
        else:
            prev_frame, prev_cx, prev_cy = track_frames[i - 1]
            dx = cx - prev_cx
            dy = cy - prev_cy
            frame_gap = frame_num - prev_frame
            if frame_gap > 0:
                speed = math.sqrt(dx*dx + dy*dy) / frame_gap
            else:
                speed = 0.0

        trajectory.append((frame_num, cx, cy, speed))

    return trajectory


def get_longest_tracks(
    predictions: Dict[int, List[Tuple]],
    top_n: int = 10
) -> List[Tuple[int, int, int, int]]:
    """Get the longest tracks (most frames).

    Args:
        predictions: Dict mapping frame_num -> tracks.
        top_n: Number of tracks to return.

    Returns:
        List of (track_id, num_frames, start_frame, end_frame) tuples.
    """
    track_info = {}  # track_id -> [frames]

    for frame_num, tracks in predictions.items():
        for track in tracks:
            track_id = track[0]
            if track_id not in track_info:
                track_info[track_id] = []
            track_info[track_id].append(frame_num)

    # Sort by number of frames
    sorted_tracks = []
    for track_id, frames in track_info.items():
        frames.sort()
        sorted_tracks.append((
            track_id,
            len(frames),
            min(frames),
            max(frames)
        ))

    sorted_tracks.sort(key=lambda x: x[1], reverse=True)

    return sorted_tracks[:top_n]
