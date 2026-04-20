#!/usr/bin/env python3
"""
Speed analysis and trajectory visualization script.

Usage:
    python scripts/analyze_speed.py --sequence MVI_39031
    python scripts/analyze_speed.py --sequence MVI_39031 --track_id 5
    python scripts/analyze_speed.py --sequence MVI_39031 --save_video
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eva import (
    Config, DetectionTrackingPipeline, UADETRACDataset,
    calculate_pixel_speed, classify_speeds, compute_track_statistics,
    analyze_speed_distribution, SpeedClass
)
from eva.speed import (
    get_track_trajectory, get_longest_tracks,
    THRESHOLD_UNIFORM, THRESHOLD_SEMANTIC, classify_speed_value
)
from eva.utils.visualization import (
    draw_track_box, draw_trajectory_with_speed,
    create_trajectory_image, VideoWriter, SPEED_CLASS_COLORS
)


def analyze_sequence(
    seq_name: str,
    config: Config,
    max_frames: int = None,
    save_video: bool = False,
    target_track_id: int = None
):
    """Analyze speed distribution for a sequence."""

    pipeline = DetectionTrackingPipeline.from_config(config)
    dataset = UADETRACDataset(config.dataset.base_path)

    print(f"Processing sequence: {seq_name}")
    print("=" * 60)

    # Process sequence
    predictions = {}
    frames_data = {}  # Store frames for visualization
    pipeline.reset_tracker()

    for frame_num, frame in dataset.iterate_frames(seq_name, max_frames):
        result = pipeline.process_frame(frame, frame_num)
        predictions[frame_num] = result.get_track_tuples(include_score=True)
        frames_data[frame_num] = frame.copy()

        if frame_num % 100 == 0:
            print(f"  Processed {frame_num} frames...")

    print(f"  Total: {len(predictions)} frames")

    # Calculate speeds
    speed_results = calculate_pixel_speed(predictions)
    distribution = analyze_speed_distribution(speed_results)

    print("\n" + "=" * 60)
    print("SPEED DISTRIBUTION")
    print("=" * 60)
    print(f"  Total samples: {distribution['count']:,}")
    print(f"  Mean: {distribution['mean']:.2f} px/frame")
    print(f"  Std: {distribution['std']:.2f}")
    print(f"  Median: {distribution['median']:.2f}")
    print(f"  Max: {distribution['max']:.2f}")

    # Analyze with both threshold options
    print("\n" + "=" * 60)
    print("SPEED CLASSIFICATION")
    print("=" * 60)

    for name, (low_th, high_th) in [
        ("Option 1: Uniform (33/33/33%)", THRESHOLD_UNIFORM),
        ("Option 2: Semantic", THRESHOLD_SEMANTIC)
    ]:
        print(f"\n[{name}]")
        print(f"  Thresholds: LOW < {low_th}, HIGH > {high_th}")

        # Classify all speeds
        all_speeds = []
        for track_id, results in speed_results.items():
            for r in results[1:]:
                if r.pixel_speed > 0:
                    all_speeds.append(r.pixel_speed)

        all_speeds = np.array(all_speeds)
        low_count = np.sum(all_speeds < low_th)
        mid_count = np.sum((all_speeds >= low_th) & (all_speeds <= high_th))
        high_count = np.sum(all_speeds > high_th)
        total = len(all_speeds)

        print(f"  저속 (LOW):    {low_count:,} ({low_count/total*100:.1f}%)")
        print(f"  중속 (MEDIUM): {mid_count:,} ({mid_count/total*100:.1f}%)")
        print(f"  고속 (HIGH):   {high_count:,} ({high_count/total*100:.1f}%)")

    # Find longest tracks
    print("\n" + "=" * 60)
    print("LONGEST TRACKS (for trajectory visualization)")
    print("=" * 60)

    longest_tracks = get_longest_tracks(predictions, top_n=10)
    for i, (track_id, num_frames, start_f, end_f) in enumerate(longest_tracks):
        print(f"  {i+1}. Track ID {track_id}: {num_frames} frames (frame {start_f} ~ {end_f})")

    # Select track for visualization
    if target_track_id is None:
        target_track_id = longest_tracks[0][0]  # Use longest track
        print(f"\n  -> Using Track ID {target_track_id} for visualization")

    # Get trajectory for selected track
    trajectory = get_track_trajectory(predictions, target_track_id)

    if len(trajectory) < 2:
        print(f"  Track {target_track_id} has insufficient data")
        return

    # Print trajectory statistics
    speeds = [s for _, _, _, s in trajectory[1:]]
    print(f"\n  Track {target_track_id} Statistics:")
    print(f"    Frames: {len(trajectory)}")
    print(f"    Avg Speed: {np.mean(speeds):.2f} px/frame")
    print(f"    Max Speed: {np.max(speeds):.2f} px/frame")
    print(f"    Min Speed: {np.min(speeds):.2f} px/frame")

    # Classify track's average speed
    avg_speed = np.mean(speeds)
    for name, (low_th, high_th) in [
        ("Uniform", THRESHOLD_UNIFORM),
        ("Semantic", THRESHOLD_SEMANTIC)
    ]:
        speed_class = classify_speed_value(avg_speed, low_th, high_th)
        print(f"    Classification ({name}): {speed_class.value}")

    # Create trajectory visualization
    output_dir = Path(config.output.base_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectory image (using middle frame as background)
    mid_frame_num = trajectory[len(trajectory)//2][0]
    if mid_frame_num in frames_data:
        bg_frame = frames_data[mid_frame_num]
    else:
        bg_frame = list(frames_data.values())[0]

    # Create images for both threshold options
    for name, (low_th, high_th) in [
        ("uniform", THRESHOLD_UNIFORM),
        ("semantic", THRESHOLD_SEMANTIC)
    ]:
        traj_img = create_trajectory_image(
            bg_frame, trajectory, target_track_id,
            low_threshold=low_th, high_threshold=high_th
        )
        img_path = output_dir / f"trajectory_{seq_name}_track{target_track_id}_{name}.jpg"
        cv2.imwrite(str(img_path), traj_img)
        print(f"\n  Saved: {img_path}")

    # Create video with trajectory if requested
    if save_video:
        video_path = output_dir / f"trajectory_{seq_name}_track{target_track_id}.mp4"
        print(f"\n  Creating video: {video_path}")

        # Get frame range for this track
        start_frame = trajectory[0][0]
        end_frame = trajectory[-1][0]

        low_th, high_th = THRESHOLD_SEMANTIC  # Use semantic for video

        with VideoWriter(str(video_path), fps=25.0) as video:
            for frame_num in range(start_frame, end_frame + 1):
                if frame_num not in frames_data:
                    continue

                frame = frames_data[frame_num].copy()

                # Draw trajectory up to current frame
                draw_trajectory_with_speed(
                    frame, trajectory, frame_num,
                    low_threshold=low_th, high_threshold=high_th,
                    thickness=2, show_points=True
                )

                # Draw current bounding box if exists
                for track in predictions.get(frame_num, []):
                    if track[0] == target_track_id:
                        box = track[1:5]
                        # Get current speed
                        current_speed = 0
                        for fn, cx, cy, spd in trajectory:
                            if fn == frame_num:
                                current_speed = spd
                                break

                        speed_class = classify_speed_value(current_speed, low_th, high_th)
                        color = SPEED_CLASS_COLORS[speed_class.name]
                        draw_track_box(frame, box, target_track_id, color=color)

                        # Show speed info
                        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
                        cv2.putText(frame, f"Track ID: {target_track_id}", (20, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Speed: {current_speed:.2f} px/f ({speed_class.value})",
                                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        break

                # Frame info
                cv2.putText(frame, f"Frame: {frame_num}", (frame.shape[1]-150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                video.write(frame)

        print(f"  Video saved!")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze speed and visualize trajectory')
    parser.add_argument('--sequence', type=str, default='MVI_39031', help='Sequence name')
    parser.add_argument('--max_frames', type=int, default=None, help='Max frames to process')
    parser.add_argument('--track_id', type=int, default=None, help='Specific track ID to visualize')
    parser.add_argument('--save_video', action='store_true', help='Save trajectory video')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    args = parser.parse_args()

    config = Config() if args.config is None else Config.from_yaml(args.config)

    analyze_sequence(
        seq_name=args.sequence,
        config=config,
        max_frames=args.max_frames,
        save_video=args.save_video,
        target_track_id=args.track_id
    )


if __name__ == "__main__":
    main()
