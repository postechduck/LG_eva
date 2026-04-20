#!/usr/bin/env python3
"""
Full evaluation script for UA-DETRAC benchmark.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --max_sequences 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eva import Config, EvaluationPipeline, load_config


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate detection and tracking on UA-DETRAC'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--max_sequences', type=int, default=None,
        help='Maximum number of sequences to evaluate'
    )
    parser.add_argument(
        '--split', type=str, default='test', choices=['train', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--conf', type=float, default=None,
        help='Confidence threshold (overrides config)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override confidence threshold if specified
    if args.conf is not None:
        config.detection.confidence_threshold = args.conf

    # Print header
    print("=" * 70)
    print("UA-DETRAC Evaluation")
    print(f"Model: {config.detection.model_type.upper()}")
    print(f"Confidence Threshold: {config.detection.confidence_threshold}")
    print("=" * 70)

    # Run evaluation
    pipeline = EvaluationPipeline(config, verbose=not args.quiet)

    det_results, track_metrics, coco_map, avg_det_latency, avg_track_latency, total_frames = \
        pipeline.run_evaluation(
            split=args.split,
            max_sequences=args.max_sequences
        )

    # Get sequence count
    if args.max_sequences:
        num_sequences = args.max_sequences
    else:
        num_sequences = len(
            pipeline.dataset.get_test_sequences()
            if args.split == 'test'
            else pipeline.dataset.get_train_sequences()
        )

    # Print results
    pipeline.print_results(
        det_results, track_metrics, coco_map,
        avg_det_latency, avg_track_latency, total_frames, num_sequences
    )


if __name__ == "__main__":
    main()
