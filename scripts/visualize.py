#!/usr/bin/env python3
"""
Visualization script for detection and tracking results.

Usage:
    python scripts/visualize.py --sequence MVI_39031
    python scripts/visualize.py --sequence MVI_39031 --mode tracking
    python scripts/visualize.py --sequence MVI_39031 --max_frames 300
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eva import (
    Config, load_config,
    UADETRACDataset,
    DetectionTrackingPipeline,
)
from eva.utils.visualization import (
    draw_detection_box,
    draw_track_box,
    draw_info_overlay,
    VideoWriter,
    VEHICLE_CLASS_NAMES,
)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize detection and tracking results'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--sequence', type=str, default='MVI_39031',
        help='Sequence name to visualize'
    )
    parser.add_argument(
        '--max_frames', type=int, default=None,
        help='Maximum number of frames to process'
    )
    parser.add_argument(
        '--mode', type=str, default='tracking',
        choices=['detection', 'tracking'],
        help='Visualization mode'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output video path (auto-generated if not specified)'
    )
    parser.add_argument(
        '--conf', type=float, default=None,
        help='Confidence threshold (overrides config)'
    )
    parser.add_argument(
        '--no_video', action='store_true',
        help='Skip video generation, show live preview only'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override confidence threshold if specified
    if args.conf is not None:
        config.detection.confidence_threshold = args.conf

    # Setup output path
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(config.output.base_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            output_dir / f'{args.mode}_{args.sequence}.mp4'
        )

    print(f"Sequence: {args.sequence}")
    print(f"Mode: {args.mode}")
    print(f"Confidence Threshold: {config.detection.confidence_threshold}")
    print(f"Output: {output_path}")
    print("=" * 70)

    # Create dataset and pipeline
    dataset = UADETRACDataset(
        base_path=config.dataset.base_path,
        image_dir=config.dataset.image_dir
    )
    pipeline = DetectionTrackingPipeline.from_config(config)

    # Get frame count
    image_dir = dataset.image_base / args.sequence
    image_files = sorted(image_dir.glob(f'*{config.dataset.image_extension}'))
    if args.max_frames:
        image_files = image_files[:args.max_frames]
    total_frames = len(image_files)

    print(f"Processing {total_frames} frames...")

    latencies = []

    with VideoWriter(output_path, fps=config.dataset.fps) as video:
        frames = dataset.iterate_frames(args.sequence, args.max_frames)

        for frame_num, frame in frames:
            # Process frame
            result = pipeline.process_frame(frame, frame_num)
            latencies.append(result.latency_ms)

            # Draw results
            if args.mode == 'detection':
                for det in result.detections:
                    draw_detection_box(
                        frame, det.box, det.class_id, det.confidence
                    )
                count = len(result.detections)
            else:
                for track in result.tracks:
                    draw_track_box(
                        frame, track.box, track.track_id,
                        show_score=True, score=track.score
                    )
                count = len(result.tracks)

            # Draw info overlay
            draw_info_overlay(
                frame,
                frame_idx=frame_num,
                total_frames=total_frames,
                detection_count=len(result.detections) if args.mode == 'detection' else 0,
                track_count=len(result.tracks) if args.mode == 'tracking' else 0,
                latency_ms=result.latency_ms,
                model_name=config.detection.model_type.upper(),
                conf_threshold=config.detection.confidence_threshold
            )

            if not args.no_video:
                video.write(frame)

            if frame_num % 50 == 0:
                print(f"  {frame_num}/{total_frames} frames processed")

    avg_latency = np.mean(latencies) if latencies else 0

    print("\nDone!")
    print(f"Output: {output_path}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"FPS: {1000/avg_latency:.1f}" if avg_latency > 0 else "FPS: N/A")


if __name__ == "__main__":
    main()
