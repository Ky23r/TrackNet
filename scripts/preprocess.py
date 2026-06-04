import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PREPROCESSING_CONFIG
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.heatmap_generator import HeatmapGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate heatmaps and train/val splits for TrackNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/images",
        help="Directory with raw game images and Label.csv files",
    )
    parser.add_argument(
        "--output-heatmaps-dir",
        type=str,
        default="data/gts",
        help="Directory to write generated Gaussian heatmap images",
    )
    parser.add_argument(
        "--output-splits-dir",
        type=str,
        default="data",
        help="Directory to write train.csv and val.csv",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=PREPROCESSING_CONFIG.train_ratio,
        help="Fraction of samples used for training",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=PREPROCESSING_CONFIG.gaussian_radius,
        help="Gaussian kernel radius in pixels",
    )
    parser.add_argument(
        "--variance",
        type=int,
        default=PREPROCESSING_CONFIG.gaussian_variance,
        help="Gaussian kernel variance",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=PREPROCESSING_CONFIG.original_image_width,
        help="Original image width (heatmap canvas size)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=PREPROCESSING_CONFIG.original_image_height,
        help="Original image height (heatmap canvas size)",
    )
    parser.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Skip heatmap generation if heatmaps already exist",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Skip CSV split creation if splits already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("TrackNet Dataset Preprocessing")
    print("=" * 60)
    print(f"Images dir  : {images_dir}")
    print(f"Heatmaps dir: {args.output_heatmaps_dir}")
    print(f"Splits dir  : {args.output_splits_dir}")
    print(f"Train ratio : {args.train_ratio}")
    print("=" * 60)

    if not args.skip_heatmaps:
        print("\n[Step 1/2] Generating Gaussian heatmaps")
        print("-" * 60)
        generator = HeatmapGenerator(
            radius=args.radius,
            variance=args.variance,
            width=args.width,
            height=args.height,
        )
        stats = generator.process_all_games(
            images_dir=images_dir,
            output_dir=args.output_heatmaps_dir,
        )
        print(f"\nHeatmaps generated: {stats['total_heatmaps']}")
    else:
        print("\n[Step 1/2] Skipping heatmap generation (--skip-heatmaps)")

    if not args.skip_splits:
        print("\n[Step 2/2] Creating train/val splits")
        print("-" * 60)
        splitter = DataSplitter(
            images_dir=images_dir,
            output_dir=args.output_splits_dir,
            train_ratio=args.train_ratio,
        )
        train_df, val_df = splitter.create_splits()
        total = len(train_df) + len(val_df)
        print(f"\nTrain: {len(train_df)} samples ({len(train_df) / total:.1%})")
        print(f"Val  : {len(val_df)} samples ({len(val_df) / total:.1%})")
    else:
        print("\n[Step 2/2] Skipping split creation (--skip-splits)")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
