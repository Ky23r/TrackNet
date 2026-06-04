import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_CONFIG, TRAINING_CONFIG
from src.datasets.tracknet_dataset import create_dataloader
from src.models.tracknet import TrackNet
from src.training.evaluator import Evaluator
from src.training.trainer import TrainingManager
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TrackNet for ball tracking in sports video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="default",
        help="Experiment name. Checkpoints → exps/<exp>/, log → logs/<exp>_*.log",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing train.csv and val.csv",
    )
    parser.add_argument(
        "--batch-size", type=int, default=TRAINING_CONFIG.default_batch_size
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG.default_epochs,
        help="Total epochs to train (not additional epochs when resuming)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=TRAINING_CONFIG.default_learning_rate,
        help="Learning rate for Adadelta optimizer",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=TRAINING_CONFIG.default_max_steps_per_epoch,
        help="Max gradient steps per epoch",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=TRAINING_CONFIG.default_validation_interval,
        help="Validate every N epochs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=TRAINING_CONFIG.default_num_workers,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda', 'cpu', or blank to auto-detect",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from (e.g. exps/my_run/last.pt)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for timestamped log files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger(args.exp, log_dir=args.log_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("TrackNet Training")
    logger.info("=" * 60)
    logger.info(f"Experiment  : {args.exp}")
    logger.info(f"Device      : {device}")
    logger.info(f"Batch size  : {args.batch_size}")
    logger.info(f"Epochs      : {args.epochs}")
    logger.info(f"LR          : {args.lr}")
    logger.info(f"Steps/epoch : {args.steps}")
    logger.info(f"Val interval: every {args.val_interval} epochs")
    logger.info(f"Data root   : {args.data_root}")
    logger.info(f"Log file    : {log_path}")
    logger.info("=" * 60)

    logger.info("[1/4] Creating dataloaders...")
    train_loader = create_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        root=args.data_root,
    )
    val_loader = create_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        root=args.data_root,
    )
    logger.info(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    logger.info("[2/4] Initializing model...")
    model = TrackNet(output_channels=MODEL_CONFIG.output_channels).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")

    logger.info("[3/4] Setting up evaluator and training manager...")
    evaluator = Evaluator(model=model, device=device)
    manager = TrainingManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        experiment_name=args.exp,
        learning_rate=args.lr,
        max_steps_per_epoch=args.steps,
        logger=logger,
    )

    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        logger.info(f"[4/4] Resuming from: {resume_path}")
        checkpoint = manager.trainer.load_checkpoint(resume_path)
        start_epoch = manager.trainer.current_epoch + 1
        if "best_f1" in checkpoint:
            manager.best_f1 = checkpoint["best_f1"]
        logger.info(
            f"Resumed from epoch {manager.trainer.current_epoch} | "
            f"Best F1 so far: {manager.best_f1:.4f}"
        )
    else:
        logger.info("[4/4] Starting fresh training run")

    logger.info("=" * 60)

    results = manager.train(
        num_epochs=args.epochs,
        validation_interval=args.val_interval,
        evaluator=evaluator,
        start_epoch=start_epoch,
    )

    logger.info("=" * 60)
    logger.info(f"Best F1     : {results['best_f1']:.4f}")
    logger.info(f"Final epoch : {results['final_epoch']}")
    logger.info(f"Checkpoints : {results['experiment_dir']}")
    logger.info(f"Log file    : {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
