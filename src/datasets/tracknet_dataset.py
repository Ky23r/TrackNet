import math
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import MODEL_CONFIG, PREPROCESSING_CONFIG


class TrackNetDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path] = PREPROCESSING_CONFIG.default_data_root,
        split: str = "train",
        height: int = MODEL_CONFIG.input_height,
        width: int = MODEL_CONFIG.input_width,
    ) -> None:
        assert split in (
            "train",
            "val",
        ), f"split must be 'train' or 'val', got '{split}'"
        self.root = Path(root)
        self.split = split
        self.height = height
        self.width = width

        csv_path = self.root / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
        row = self.df.iloc[idx]

        img_path = self.root / row["image_path"]
        prev_path = self.root / row["prev_image_path"]
        prev2_path = self.root / row["prev2_image_path"]
        gt_path = self.root / row["gt_path"]

        x_coord = row["x-coordinate"]
        y_coord = row["y-coordinate"]
        x = float(x_coord) if not math.isnan(x_coord) else -1.0
        y = float(y_coord) if not math.isnan(y_coord) else -1.0
        visibility = int(row["visibility"])

        frames = self._load_and_stack_frames(img_path, prev_path, prev2_path)
        heatmap = self._load_heatmap(gt_path)

        return frames, heatmap, x, y, visibility

    def _load_heatmap(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load heatmap: {path}")
        img = cv2.resize(img, (self.width, self.height))
        return img.reshape(-1)

    def _load_and_stack_frames(
        self, current: Path, prev: Path, prev2: Path
    ) -> np.ndarray:
        frames = []
        for path in (current, prev, prev2):
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Could not load frame: {path}")
            img = cv2.resize(img, (self.width, self.height))
            frames.append(img)

        stacked = np.concatenate(frames, axis=2).astype(np.float32) / 255.0
        return np.moveaxis(stacked, 2, 0)

    def get_sample_info(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        x_val = row["x-coordinate"]
        y_val = row["y-coordinate"]
        return {
            "index": idx,
            "image_path": row["image_path"],
            "x": None if math.isnan(x_val) else float(x_val),
            "y": None if math.isnan(y_val) else float(y_val),
            "visibility": int(row["visibility"]),
            "status": row.get("status", "unknown"),
        }


def create_dataloader(
    split: str,
    batch_size: int,
    num_workers: int = 1,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    root: Union[str, Path] = PREPROCESSING_CONFIG.default_data_root,
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"

    dataset = TrackNetDataset(root=root, split=split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
