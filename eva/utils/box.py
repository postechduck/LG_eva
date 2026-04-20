"""Bounding box utilities for EVA.

Contains IoU computation, ignored region handling, and box filtering functions.
"""

from typing import List, Tuple, Sequence

# Box type: (x1, y1, x2, y2)
Box = Tuple[float, float, float, float]


def compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute Intersection over Union (IoU) between two boxes.

    Args:
        box1: First box as (x1, y1, x2, y2).
        box2: Second box as (x1, y1, x2, y2).

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def is_in_ignored_region(
    box: Sequence[float],
    ignored_regions: List[Box]
) -> bool:
    """Check if box center is inside any ignored region.

    Args:
        box: Box as (x1, y1, x2, y2).
        ignored_regions: List of ignored region boxes as (x1, y1, x2, y2).

    Returns:
        True if box center is inside any ignored region.
    """
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2

    for ig_x1, ig_y1, ig_x2, ig_y2 in ignored_regions:
        if ig_x1 <= center_x <= ig_x2 and ig_y1 <= center_y <= ig_y2:
            return True

    return False


def filter_ignored_boxes(
    boxes: List[Tuple],
    ignored_regions: List[Box],
    box_index: int = 0
) -> List[Tuple]:
    """Filter out boxes whose centers are in ignored regions.

    Args:
        boxes: List of boxes/tuples. Each item should contain box coords.
        ignored_regions: List of ignored region boxes.
        box_index: Start index of box coordinates in each tuple.
            If box_index=0, expects (x1, y1, x2, y2, ...).
            If box_index=1, expects (id, x1, y1, x2, y2, ...).

    Returns:
        Filtered list of boxes.
    """
    if not ignored_regions:
        return boxes

    filtered = []
    for item in boxes:
        box = item[box_index:box_index + 4]
        if not is_in_ignored_region(box, ignored_regions):
            filtered.append(item)

    return filtered
