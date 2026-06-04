from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from ..config import PATH_CONFIG, PREPROCESSING_CONFIG


class DataSplitter:
    def __init__(
        self,
        images_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        train_ratio: float = PREPROCESSING_CONFIG.train_ratio,
        random_seed: int = PREPROCESSING_CONFIG.random_seed,
    ) -> None:
        if images_dir is None:
            images_dir = (
                Path(PREPROCESSING_CONFIG.default_data_root)
                / PREPROCESSING_CONFIG.images_subdir
            )
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir or PREPROCESSING_CONFIG.default_data_root)
        self.train_ratio = train_ratio
        self.random_seed = random_seed

    def create_splits(
        self,
        game_id_start: int = PREPROCESSING_CONFIG.game_id_start,
        game_id_end: int = PREPROCESSING_CONFIG.game_id_end,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_labels = pd.DataFrame()

        for game_id in range(game_id_start, game_id_end):
            game_name = f"game{game_id}"
            game_path = self.images_dir / game_name

            if not game_path.exists():
                print(f"  Skipping {game_name} (not found)")
                continue

            print(f"  Processing {game_name}")
            clips = sorted(p.name for p in game_path.iterdir() if p.is_dir())

            for clip in clips:
                labels_path = game_path / clip / PATH_CONFIG.label_csv_filename
                if not labels_path.exists():
                    continue
                print(f"    {clip}")
                clip_df = self._process_clip(game_name, clip, labels_path)
                all_labels = pd.concat([all_labels, clip_df], ignore_index=True)

        print(f"Total samples: {len(all_labels)}")
        all_labels = all_labels.sample(
            frac=1, random_state=self.random_seed
        ).reset_index(drop=True)

        n_train = int(len(all_labels) * self.train_ratio)
        train_df = all_labels[:n_train]
        val_df = all_labels[n_train:]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.output_dir / PATH_CONFIG.train_csv_filename
        val_path = self.output_dir / PATH_CONFIG.val_csv_filename

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(f"Saved: {train_path} ({len(train_df)} samples)")
        print(f"Saved: {val_path} ({len(val_df)} samples)")

        return train_df, val_df

    def _process_clip(
        self, game_name: str, clip: str, labels_path: Path
    ) -> pd.DataFrame:
        df = pd.read_csv(labels_path)

        df["gt_path"] = (
            f"{PREPROCESSING_CONFIG.heatmaps_subdir}/{game_name}/{clip}/"
            + df["file name"]
        )
        df["image_path"] = (
            f"{PREPROCESSING_CONFIG.images_subdir}/{game_name}/{clip}/"
            + df["file name"]
        )

        temporal_df = df[2:].copy()
        temporal_df.loc[:, "prev_image_path"] = list(df["image_path"][1:-1])
        temporal_df.loc[:, "prev2_image_path"] = list(df["image_path"][:-2])

        return temporal_df

    def get_split_statistics(self) -> Dict[str, object]:
        train_path = self.output_dir / PATH_CONFIG.train_csv_filename
        val_path = self.output_dir / PATH_CONFIG.val_csv_filename

        if not train_path.exists() or not val_path.exists():
            return {
                "exists": False,
                "message": "Split files not found. Run create_splits() first.",
            }

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        total = len(train_df) + len(val_df)

        return {
            "exists": True,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "total_samples": total,
            "train_ratio": len(train_df) / total,
            "train_visibility_distribution": train_df["visibility"]
            .value_counts()
            .to_dict(),
            "val_visibility_distribution": val_df["visibility"]
            .value_counts()
            .to_dict(),
        }
