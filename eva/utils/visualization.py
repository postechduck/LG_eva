"""Visualization utilities for EVA.

Contains functions for drawing bounding boxes, tracks, and creating videos.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# COCO vehicle class names and colors
VEHICLE_CLASS_NAMES: Dict[int, str] = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

VEHICLE_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    2: (0, 255, 0),    # car - green
    3: (255, 0, 0),    # motorcycle - blue
    5: (0, 165, 255),  # bus - orange
    7: (0, 0, 255)     # truck - red
}


def get_track_color(track_id: int, seed: int = 42) -> Tuple[int, int, int]:
    """Get a consistent color for a track ID.

    Args:
        track_id: Track identifier.
        seed: Random seed for color generation.

    Returns:
        BGR color tuple.
    """
    np.random.seed(seed)
    colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
    return tuple(int(c) for c in colors[track_id % 1000])


def draw_detection_box(
    frame: np.ndarray,
    box: Tuple[float, float, float, float],
    class_id: int,
    confidence: float,
    color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """Draw a detection bounding box with label on frame.

    Args:
        frame: Image frame (BGR).
        box: Bounding box as (x1, y1, x2, y2).
        class_id: COCO class ID.
        confidence: Detection confidence score.
        color: Optional custom color. Uses class color if not provided.

    Returns:
        Frame with drawn box.
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    if color is None:
        color = VEHICLE_CLASS_COLORS.get(class_id, (255, 255, 255))

    class_name = VEHICLE_CLASS_NAMES.get(class_id, f'cls_{class_id}')
    label = f"{class_name} {confidence:.2f}"

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    (label_w, label_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1
    )

    # Draw label text
    cv2.putText(
        frame, label, (x1, y1 - 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    return frame


def draw_track_box(
    frame: np.ndarray,
    box: Tuple[float, float, float, float],
    track_id: int,
    color: Optional[Tuple[int, int, int]] = None,
    show_score: bool = False,
    score: float = 0.0
) -> np.ndarray:
    """Draw a tracking bounding box with ID on frame.

    Args:
        frame: Image frame (BGR).
        box: Bounding box as (x1, y1, x2, y2).
        track_id: Track identifier.
        color: Optional custom color. Uses track-based color if not provided.
        show_score: Whether to show confidence score.
        score: Confidence score (only used if show_score is True).

    Returns:
        Frame with drawn box.
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    if color is None:
        color = get_track_color(track_id)

    label = f"ID:{track_id}"
    if show_score:
        label += f" {score:.2f}"

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    (label_w, label_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1
    )

    # Draw label text
    cv2.putText(
        frame, label, (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    return frame


def draw_info_overlay(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    detection_count: int = 0,
    track_count: int = 0,
    latency_ms: float = 0.0,
    model_name: str = "RT-DETR-L",
    conf_threshold: float = 0.3
) -> np.ndarray:
    """Draw information overlay on frame.

    Args:
        frame: Image frame (BGR).
        frame_idx: Current frame index (1-based).
        total_frames: Total number of frames.
        detection_count: Number of detections in frame.
        track_count: Number of tracks in frame.
        latency_ms: Detection latency in milliseconds.
        model_name: Name of the detection model.
        conf_threshold: Confidence threshold used.

    Returns:
        Frame with info overlay.
    """
    h, w = frame.shape[:2]

    # Left info
    info_text = f"Frame: {frame_idx}/{total_frames}"
    if detection_count > 0:
        info_text += f" | Det: {detection_count}"
    if track_count > 0:
        info_text += f" | Tracks: {track_count}"
    if latency_ms > 0:
        info_text += f" | {latency_ms:.1f}ms"

    cv2.rectangle(frame, (0, 0), (500, 30), (0, 0, 0), -1)
    cv2.putText(
        frame, info_text, (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )

    # Right info
    model_text = f"{model_name} | conf={conf_threshold}"
    cv2.rectangle(frame, (w - 220, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(
        frame, model_text, (w - 210, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
    )

    return frame


class VideoWriter:
    """Context manager for writing video files."""

    def __init__(
        self,
        output_path: str,
        fps: float = 25.0,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = 'mp4v'
    ):
        """Initialize video writer.

        Args:
            output_path: Output video file path.
            fps: Frames per second.
            frame_size: (width, height). Can be set later from first frame.
            codec: FourCC codec string.
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.release()
        return False

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to video.

        Args:
            frame: BGR image frame.
        """
        if self.writer is None:
            h, w = frame.shape[:2]
            if self.frame_size is None:
                self.frame_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, self.frame_size
            )

        self.writer.write(frame)


# Speed class colors
SPEED_CLASS_COLORS = {
    'LOW': (0, 255, 0),      # Green - 저속
    'MEDIUM': (0, 255, 255), # Yellow - 중속
    'HIGH': (0, 0, 255),     # Red - 고속
}


def draw_trajectory(
    frame: np.ndarray,
    trajectory: List[Tuple[int, float, float, float]],
    current_frame: int,
    color: Tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
    show_points: bool = True,
    fade_old: bool = True
) -> np.ndarray:
    """Draw trajectory line on frame.

    Args:
        frame: BGR image frame.
        trajectory: List of (frame_num, cx, cy, speed) from get_track_trajectory.
        current_frame: Current frame number (only draw up to this frame).
        color: Line color (BGR).
        thickness: Line thickness.
        show_points: Whether to draw points at each position.
        fade_old: Whether to fade older points.

    Returns:
        Frame with trajectory drawn.
    """
    # Filter trajectory up to current frame
    points = [(cx, cy) for fn, cx, cy, _ in trajectory if fn <= current_frame]

    if len(points) < 2:
        if len(points) == 1 and show_points:
            pt = (int(points[0][0]), int(points[0][1]))
            cv2.circle(frame, pt, 5, color, -1)
        return frame

    # Draw trajectory line
    for i in range(1, len(points)):
        pt1 = (int(points[i-1][0]), int(points[i-1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))

        if fade_old:
            # Fade older segments
            alpha = 0.3 + 0.7 * (i / len(points))
            line_color = tuple(int(c * alpha) for c in color)
        else:
            line_color = color

        cv2.line(frame, pt1, pt2, line_color, thickness)

    # Draw points
    if show_points:
        for i, (cx, cy) in enumerate(points):
            pt = (int(cx), int(cy))
            if fade_old:
                alpha = 0.3 + 0.7 * (i / len(points))
                pt_color = tuple(int(c * alpha) for c in color)
            else:
                pt_color = color

            radius = 3 if i < len(points) - 1 else 6
            cv2.circle(frame, pt, radius, pt_color, -1)

    return frame


def draw_trajectory_with_speed(
    frame: np.ndarray,
    trajectory: List[Tuple[int, float, float, float]],
    current_frame: int,
    low_threshold: float = 1.0,
    high_threshold: float = 5.0,
    thickness: int = 2,
    show_points: bool = True
) -> np.ndarray:
    """Draw trajectory with speed-based coloring.

    Args:
        frame: BGR image frame.
        trajectory: List of (frame_num, cx, cy, speed).
        current_frame: Current frame number.
        low_threshold: Speed threshold for LOW.
        high_threshold: Speed threshold for HIGH.
        thickness: Line thickness.
        show_points: Whether to draw points.

    Returns:
        Frame with colored trajectory.
    """
    # Filter trajectory up to current frame
    filtered = [(cx, cy, spd) for fn, cx, cy, spd in trajectory if fn <= current_frame]

    if len(filtered) < 2:
        if len(filtered) == 1:
            pt = (int(filtered[0][0]), int(filtered[0][1]))
            cv2.circle(frame, pt, 5, SPEED_CLASS_COLORS['LOW'], -1)
        return frame

    # Draw trajectory segments with speed colors
    for i in range(1, len(filtered)):
        cx1, cy1, _ = filtered[i-1]
        cx2, cy2, speed = filtered[i]

        pt1 = (int(cx1), int(cy1))
        pt2 = (int(cx2), int(cy2))

        # Determine color based on speed
        if speed < low_threshold:
            color = SPEED_CLASS_COLORS['LOW']
        elif speed > high_threshold:
            color = SPEED_CLASS_COLORS['HIGH']
        else:
            color = SPEED_CLASS_COLORS['MEDIUM']

        cv2.line(frame, pt1, pt2, color, thickness)

        if show_points:
            cv2.circle(frame, pt2, 3, color, -1)

    # Draw start point
    start_pt = (int(filtered[0][0]), int(filtered[0][1]))
    cv2.circle(frame, start_pt, 6, (255, 255, 255), -1)
    cv2.circle(frame, start_pt, 6, (0, 0, 0), 2)

    # Draw current point (larger)
    end_pt = (int(filtered[-1][0]), int(filtered[-1][1]))
    cv2.circle(frame, end_pt, 8, (255, 0, 255), -1)

    return frame


def create_trajectory_image(
    background: np.ndarray,
    trajectory: List[Tuple[int, float, float, float]],
    track_id: int,
    low_threshold: float = 1.0,
    high_threshold: float = 5.0
) -> np.ndarray:
    """Create a single image showing full trajectory.

    Args:
        background: Background image (first or middle frame).
        trajectory: Full trajectory data.
        track_id: Track ID for labeling.
        low_threshold: Speed threshold for LOW.
        high_threshold: Speed threshold for HIGH.

    Returns:
        Image with full trajectory drawn.
    """
    frame = background.copy()

    # Draw full trajectory
    draw_trajectory_with_speed(
        frame, trajectory, trajectory[-1][0],
        low_threshold, high_threshold,
        thickness=3, show_points=True
    )

    # Add legend
    h, w = frame.shape[:2]
    legend_y = 50
    cv2.rectangle(frame, (w-200, 10), (w-10, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Track ID: {track_id}", (w-190, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    legend_y += 25
    cv2.circle(frame, (w-180, legend_y-5), 5, SPEED_CLASS_COLORS['LOW'], -1)
    cv2.putText(frame, f"LOW (<{low_threshold})", (w-165, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    legend_y += 20
    cv2.circle(frame, (w-180, legend_y-5), 5, SPEED_CLASS_COLORS['MEDIUM'], -1)
    cv2.putText(frame, f"MED ({low_threshold}-{high_threshold})", (w-165, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    legend_y += 20
    cv2.circle(frame, (w-180, legend_y-5), 5, SPEED_CLASS_COLORS['HIGH'], -1)
    cv2.putText(frame, f"HIGH (>{high_threshold})", (w-165, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Add statistics
    speeds = [s for _, _, _, s in trajectory[1:]]
    if speeds:
        avg_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)

        cv2.rectangle(frame, (10, h-70), (250, h-10), (0, 0, 0), -1)
        cv2.putText(frame, f"Frames: {len(trajectory)}", (20, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Avg Speed: {avg_speed:.2f} px/f", (20, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame
