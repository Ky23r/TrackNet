import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TRAINING_CONFIG
from src.datasets.tracknet_dataset import create_dataloader
from src.models.tracknet import load_model
from src.training.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TrackNet on the validation set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to checkpoint (e.g. exps/my_run/best.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing val.csv",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--workers", type=int, default=TRAINING_CONFIG.default_num_workers
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda', 'cpu', or blank to auto-detect",
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show per-visibility-class breakdown"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)

    print("=" * 60)
    print("TrackNet Evaluation")
    print("=" * 60)
    print(f"Model     : {model_path}")
    print(f"Device    : {device}")
    print(f"Data root : {args.data_root}")
    print("=" * 60)

    print("\n[1/3] Creating dataloader...")
    val_loader = create_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        root=args.data_root,
    )
    print(f"Validation batches: {len(val_loader)}")

    print("\n[2/3] Loading model...")
    try:
        model = load_model(model_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model loaded successfully")

    print("\n[3/3] Running evaluation...")
    evaluator = Evaluator(model=model, device=device)
    metrics = evaluator.evaluate(val_loader)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Loss      : {metrics['loss']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print("-" * 60)
    print(f"TP : {metrics['true_positives']}")
    print(f"FP : {metrics['false_positives']}")
    print(f"TN : {metrics['true_negatives']}")
    print(f"FN : {metrics['false_negatives']}")
    print("=" * 60)

    if args.detailed:
        print("\n" + "=" * 60)
        print("Per-Visibility-Class Metrics")
        print("=" * 60)

        class_metrics = evaluator.compute_class_wise_metrics(val_loader)

        visibility_names = {
            0: "Class 0 — Not Visible",
            1: "Class 1 — Visible",
            2: "Class 2 — Visible (partial)",
            3: "Class 3 — Visible (motion blur)",
        }

        for vis_class, m in class_metrics.items():
            print(f"\n{visibility_names.get(vis_class, f'Class {vis_class}')}")
            print("-" * 60)
            print(f"  Samples   : {m['total_samples']}")
            print(f"  Precision : {m['precision']:.4f}")
            print(f"  Recall    : {m['recall']:.4f}")
            print(f"  F1 Score  : {m['f1_score']:.4f}")
            print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")

        print("=" * 60)


if __name__ == "__main__":
    main()
