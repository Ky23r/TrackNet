import math
import os

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TrackNetDataset(Dataset):
    def __init__(self, root="data", split="train", height=360, width=640):
        self.root = root
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        self.df = pd.read_csv(os.path.join(self.root, f"{split}.csv"))
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["image_path"])
        prev_path = os.path.join(self.root, row["prev_image_path"])
        prev2_path = os.path.join(self.root, row["prev2_image_path"])
        gt_path = os.path.join(self.root, row["gt_path"])

        x = row["x-coordinate"] if not math.isnan(row["x-coordinate"]) else -1
        y = row["y-coordinate"] if not math.isnan(row["y-coordinate"]) else -1
        vis = row["visibility"]

        frames = self._load_frames(img_path, prev_path, prev2_path)
        heatmap = self._load_heatmap(gt_path)

        return frames, heatmap, x, y, vis

    def _load_heatmap(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))
        img = img[:, :, 0]
        return img.reshape(-1)

    def _load_frames(self, cur, prev, prev2):
        frames = []
        for path in [cur, prev, prev2]:
            img = cv2.imread(path)
            img = cv2.resize(img, (self.width, self.height))
            frames.append(img)

        frames = np.concatenate(frames, axis=2)
        frames = frames.astype(np.float32) / 255.0
        frames = np.moveaxis(frames, 2, 0)
        return frames


def main():
    dataset = TrackNetDataset(root="data", split="train", height=360, width=640)
    frames, heatmap, x, y, vis = dataset[0]
    print(f"Frames: {frames.shape}")
    print(f"Heatmap: {heatmap.shape}")
    print(f"Position: ({x}, {y})")
    print(f"Visibility: {vis}")


if __name__ == "__main__":
    main()
