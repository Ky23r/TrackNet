import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import INFERENCE_CONFIG, MODEL_CONFIG
from src.inference.video_inferencer import VideoInferencer
from src.models.tracknet import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TrackNet ball-tracking inference on a video file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to checkpoint (e.g. exps/my_run/best.pt or pretrained/best.pt)",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input video file path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output video file path"
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Fill short trajectory gaps with linear interpolation",
    )
    parser.add_argument(
        "--trace-length",
        type=int,
        default=INFERENCE_CONFIG.default_trace_length,
        help="Frames of trajectory trail to draw",
    )
    parser.add_argument(
        "--width", type=int, default=MODEL_CONFIG.input_width, help="Model input width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=MODEL_CONFIG.input_height,
        help="Model input height",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda', 'cpu', or blank to auto-detect",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input video not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TrackNet Video Inference")
    print("=" * 60)
    print(f"Model       : {model_path}")
    print(f"Input       : {input_path}")
    print(f"Output      : {output_path}")
    print(f"Device      : {device}")
    print(f"Interpolate : {'yes' if args.interpolate else 'no'}")
    print(f"Trace length: {args.trace_length}")
    print("=" * 60)

    print("\n[1/2] Loading model...")
    try:
        model = load_model(model_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model loaded successfully")

    print("\n[2/2] Setting up inferencer...")
    inferencer = VideoInferencer(
        model=model,
        device=device,
        input_width=args.width,
        input_height=args.height,
    )

    print("\n" + "=" * 60)
    print("Running Inference")
    print("=" * 60 + "\n")

    results = inferencer.infer_on_video(
        video_path=str(input_path),
        output_path=str(output_path),
        enable_interpolation=args.interpolate,
        trace_length=args.trace_length,
    )

    print("\n" + "=" * 60)
    print("Inference Complete")
    print("=" * 60)
    print(f"Total frames   : {results['total_frames']}")
    print(f"Detected frames: {results['detected_frames']}")
    print(f"Detection rate : {results['detection_rate']:.2%}")
    print(f"FPS            : {results['fps']}")
    print(f"Resolution     : {results['resolution'][0]}x{results['resolution'][1]}")
    print(f"Output saved to: {results['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
