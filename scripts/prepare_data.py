#!/usr/bin/env python
"""
Data preparation script.

Usage:
    python scripts/prepare_data.py --dataset hateful_memes --output data/
    python scripts/prepare_data.py --dataset mmhs150k --raw_dir data/_raw_mmhs/MMHS150K --output data/mmhs150k
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    download_and_prepare_hateful_memes,
    prepare_mmhs150k_from_raw,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hateful_memes", "mmhs150k"],
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="Raw data directory (required for mmhs150k)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    
    if args.dataset == "hateful_memes":
        print("Preparing Hateful Memes dataset...")
        download_and_prepare_hateful_memes(output_dir)
        print("Done!")
        
    elif args.dataset == "mmhs150k":
        if not args.raw_dir:
            print("Error: --raw_dir is required for mmhs150k dataset")
            sys.exit(1)
        
        raw_dir = Path(args.raw_dir)
        if not raw_dir.exists():
            print(f"Error: Raw directory not found: {raw_dir}")
            sys.exit(1)
        
        print("Preparing MMHS150K dataset...")
        prepare_mmhs150k_from_raw(raw_dir, output_dir)
        print("Done!")


if __name__ == "__main__":
    main()
