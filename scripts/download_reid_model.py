#!/usr/bin/env python3
"""Download Re-ID model for DeepSORT tracker.

Downloads Market1501 trained Re-ID model weights.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_reid_model():
    """Download Re-ID model weights."""
    import gdown

    output_dir = PROJECT_ROOT / "models" / "tracking" / "reid"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ckpt.t7"

    if output_path.exists():
        print(f"Re-ID model already exists at: {output_path}")
        return str(output_path)

    # Market1501 trained Re-ID model (512-dim feature)
    # Source: https://github.com/ZQPei/deep_sort_pytorch
    url = "https://drive.google.com/uc?id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN"

    print(f"Downloading Re-ID model to: {output_path}")
    gdown.download(url, str(output_path), quiet=False)

    print(f"Download complete: {output_path}")
    return str(output_path)


def main():
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown

    model_path = download_reid_model()
    print(f"\nRe-ID model ready at: {model_path}")
    print("\nYou can now use DeepSORT with:")
    print("  python run.py --config configs/deepsort.yaml")


if __name__ == "__main__":
    main()
