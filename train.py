import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TrackNetDataset
from model import TrackNet
from utils import train_epoch, validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--exp", type=str, default="default")
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    train_set = TrackNetDataset(split="train")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_set = TrackNetDataset(split="val")
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrackNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    exp_dir = f"exps/{args.exp}"
    os.makedirs(exp_dir, exist_ok=True)

    best_path = os.path.join(exp_dir, "best.pt")
    last_path = os.path.join(exp_dir, "last.pt")
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, args.steps
        )
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}")

        if epoch % args.val_interval == 0:
            val_loss, precision, recall, f1 = validate(model, val_loader, device, epoch)
            print(
                f"Val Loss: {val_loss:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)
                print(f"Best model saved (F1: {best_f1:.4f})")

            torch.save(model.state_dict(), last_path)


if __name__ == "__main__":
    main()

