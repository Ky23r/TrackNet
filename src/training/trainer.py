import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..config import PATH_CONFIG, TRAINING_CONFIG


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        learning_rate: float = TRAINING_CONFIG.default_learning_rate,
        max_steps_per_epoch: int = TRAINING_CONFIG.default_max_steps_per_epoch,
    ) -> None:
        self.model = model
        self.device = device
        self.max_steps_per_epoch = max_steps_per_epoch

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.current_epoch: int = 0
        self.training_losses: List[float] = []

    def train_epoch(
        self, train_loader: torch.utils.data.DataLoader, epoch: int
    ) -> float:
        self.current_epoch = epoch
        epoch_losses: List[float] = []

        self.model.train()

        total_steps = min(self.max_steps_per_epoch, len(train_loader))
        pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=total_steps)

        for batch_idx, batch in pbar:
            if batch_idx >= self.max_steps_per_epoch:
                break
            loss = self._train_step(batch)
            epoch_losses.append(loss)
            pbar.set_postfix(loss=f"{loss:.4f}")

        avg_loss = float(np.mean(epoch_losses))
        self.training_losses.append(avg_loss)
        return avg_loss

    def _train_step(self, batch: Any) -> float:
        self.optimizer.zero_grad()

        frames = batch[0].float().to(self.device)
        target_heatmap = batch[1].long().to(self.device)

        output = self.model(frames, inference=False)
        loss = self.criterion(output, target_heatmap)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(
        self, save_path: Path, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "training_losses": self.training_losses,
        }

        if additional_info is not None:
            checkpoint.update(additional_info)

        torch.save(checkpoint, save_path)

    def load_checkpoint(
        self, checkpoint_path: Path, load_optimizer: bool = True
    ) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if load_optimizer and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_epoch = checkpoint.get("epoch", 0)
            self.training_losses = checkpoint.get("training_losses", [])
        else:
            self.model.load_state_dict(checkpoint)

        return checkpoint


class TrainingManager:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str,
        experiment_name: str = "default",
        learning_rate: float = TRAINING_CONFIG.default_learning_rate,
        max_steps_per_epoch: int = TRAINING_CONFIG.default_max_steps_per_epoch,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.trainer = Trainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            max_steps_per_epoch=max_steps_per_epoch,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_name = experiment_name
        self.logger = logger or logging.getLogger(__name__)

        self.exp_dir = Path(PATH_CONFIG.experiments_dir) / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.best_f1: float = 0.0

    @property
    def _best_path(self) -> Path:
        return self.exp_dir / PATH_CONFIG.best_model_filename

    @property
    def _last_path(self) -> Path:
        return self.exp_dir / PATH_CONFIG.last_model_filename

    def _save(self, path: Path) -> None:
        self.trainer.save_checkpoint(path, {"best_f1": self.best_f1})

    def train(
        self,
        num_epochs: int,
        validation_interval: int = TRAINING_CONFIG.default_validation_interval,
        evaluator: Optional[Any] = None,
        start_epoch: int = 1,
    ) -> Dict[str, Any]:
        if start_epoch > num_epochs:
            self.logger.warning(
                f"start_epoch ({start_epoch}) > num_epochs ({num_epochs}). "
                f"Nothing to train. Increase --epochs to continue."
            )
            return {
                "best_f1": self.best_f1,
                "final_epoch": self.trainer.current_epoch,
                "experiment_dir": str(self.exp_dir),
            }

        self.logger.info(f"Experiment : {self.experiment_name}")
        self.logger.info(f"Device     : {self.trainer.device}")
        self.logger.info(
            f"Training   : epochs {start_epoch}–{num_epochs}, steps/epoch {self.trainer.max_steps_per_epoch}"
        )
        self.logger.info(f"Output     : {self.exp_dir}")

        current_epoch = start_epoch - 1

        try:
            for epoch in range(start_epoch, num_epochs + 1):
                current_epoch = epoch
                train_loss = self.trainer.train_epoch(self.train_loader, epoch)
                self.logger.info(
                    f"Epoch {epoch}/{num_epochs} | train_loss={train_loss:.4f}"
                )

                self._save(self._last_path)

                if epoch % validation_interval == 0 and evaluator is not None:
                    val_metrics = evaluator.evaluate(self.val_loader, epoch=epoch)
                    self.logger.info(
                        f"Epoch {epoch}/{num_epochs} | "
                        f"val_loss={val_metrics['loss']:.4f} | "
                        f"P={val_metrics['precision']:.4f} "
                        f"R={val_metrics['recall']:.4f} "
                        f"F1={val_metrics['f1_score']:.4f}"
                    )

                    if val_metrics["f1_score"] > self.best_f1:
                        self.best_f1 = val_metrics["f1_score"]
                        self._save(self._best_path)
                        self.logger.info(
                            f"New best F1={self.best_f1:.4f} → saved best.pt"
                        )

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user (Ctrl+C).")
            if current_epoch >= start_epoch:
                self.logger.info(
                    f"Saving checkpoint at epoch {current_epoch} to last.pt..."
                )
                self._save(self._last_path)
                self.logger.info(f"Resume with: --resume {self._last_path}")

        self.logger.info("-" * 50)
        self.logger.info("Training complete.")
        self.logger.info(f"Best F1   : {self.best_f1:.4f}")
        self.logger.info(f"Output    : {self.exp_dir}")

        return {
            "best_f1": self.best_f1,
            "final_epoch": current_epoch,
            "experiment_dir": str(self.exp_dir),
        }
