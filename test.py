import argparse

import torch
from torch.utils.data import DataLoader

from dataset import TrackNetDataset
from model import TrackNet
from utils import validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrackNet()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)

    # Setup validation dataset
    val_set = TrackNetDataset(split="val")
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Evaluate
    val_loss, precision, recall, f1 = validate(model, val_loader, device, -1)

    print(f"Validation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
