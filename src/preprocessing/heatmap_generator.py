from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np
import pandas as pd

from ..config import PATH_CONFIG, PREPROCESSING_CONFIG


class HeatmapGenerator:
    def __init__(
        self,
        radius: int = PREPROCESSING_CONFIG.gaussian_radius,
        variance: int = PREPROCESSING_CONFIG.gaussian_variance,
        width: int = PREPROCESSING_CONFIG.original_image_width,
        height: int = PREPROCESSING_CONFIG.original_image_height,
    ) -> None:
        self.radius = radius
        self.variance = variance
        self.width = width
        self.height = height
        self.kernel = self._create_gaussian_kernel()

    def _create_gaussian_kernel(self) -> np.ndarray:
        x, y = np.mgrid[-self.radius : self.radius + 1, -self.radius : self.radius + 1]
        kernel = np.exp(-(x**2 + y**2) / (2 * self.variance))
        kernel = kernel * 255 / kernel[self.radius, self.radius]
        return kernel.astype(int)

    def generate_heatmap(self, x: int, y: int, visibility: int) -> np.ndarray:
        heatmap = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if visibility == 0:
            return heatmap

        x0 = max(0, x - self.radius)
        x1 = min(self.width, x + self.radius + 1)
        y0 = max(0, y - self.radius)
        y1 = min(self.height, y + self.radius + 1)

        kx0 = x0 - (x - self.radius)
        ky0 = y0 - (y - self.radius)
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)

        weights = self.kernel[ky0:ky1, kx0:kx1].clip(0, 255).astype(np.uint8)
        heatmap[y0:y1, x0:x1] = weights[:, :, np.newaxis]

        return heatmap

    def generate_heatmaps_for_clip(
        self,
        labels_df: pd.DataFrame,
        output_dir: Union[str, Path],
    ) -> int:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for _, row in labels_df.iterrows():
            visibility = row["visibility"]
            x = int(row["x-coordinate"]) if visibility != 0 else 0
            y = int(row["y-coordinate"]) if visibility != 0 else 0
            heatmap = self.generate_heatmap(x, y, visibility)
            cv2.imwrite(str(output_dir / row["file name"]), heatmap)
            count += 1

        return count

    def process_all_games(
        self,
        images_dir: Union[str, Path],
        output_dir: Union[str, Path],
        game_id_start: int = PREPROCESSING_CONFIG.game_id_start,
        game_id_end: int = PREPROCESSING_CONFIG.game_id_end,
    ) -> Dict[str, object]:
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)

        print("Generating heatmaps...")
        total = 0
        game_stats: Dict[str, int] = {}

        for game_id in range(game_id_start, game_id_end):
            game_name = f"game{game_id}"
            game_path = images_dir / game_name

            if not game_path.exists():
                print(f"  Skipping {game_name} (not found)")
                continue

            print(f"  Processing {game_name}")
            clips = sorted(p.name for p in game_path.iterdir() if p.is_dir())
            game_count = 0

            for clip in clips:
                labels_path = game_path / clip / PATH_CONFIG.label_csv_filename
                if not labels_path.exists():
                    continue
                print(f"    {clip}")
                labels_df = pd.read_csv(labels_path)
                game_count += self.generate_heatmaps_for_clip(
                    labels_df, output_dir / game_name / clip
                )

            game_stats[game_name] = game_count
            total += game_count

        print(f"Generated {total} heatmaps across {len(game_stats)} games")
        return {"total_heatmaps": total, "game_stats": game_stats}
