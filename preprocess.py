import os

import cv2
import numpy as np
import pandas as pd


def gaussian_kernel(radius, var):
    x, y = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x**2 + y**2) / (2 * var))
    kernel = kernel * 255 / kernel[radius, radius]
    return kernel.astype(int)


def generate_heatmaps(img_dir, out_dir, radius, var, width, height):
    print("Generating heatmaps...")
    kernel = gaussian_kernel(radius, var)

    for game_id in range(1, 11):
        game_dir = f"game{game_id}"
        clips = os.listdir(os.path.join(img_dir, game_dir))
        print(f"  {game_dir}")

        for clip in clips:
            print(f"    {clip}")

            out_game_dir = os.path.join(out_dir, game_dir)
            os.makedirs(out_game_dir, exist_ok=True)

            out_clip_dir = os.path.join(out_game_dir, clip)
            os.makedirs(out_clip_dir, exist_ok=True)

            labels_path = os.path.join(img_dir, game_dir, clip, "Label.csv")
            df = pd.read_csv(labels_path)

            for idx in range(len(df)):
                fname, vis, x, y, _ = df.iloc[idx]
                heatmap = np.zeros((height, width, 3), dtype=np.uint8)

                if vis != 0:
                    x, y = int(x), int(y)
                    for i in range(-radius, radius + 1):
                        for j in range(-radius, radius + 1):
                            px, py = x + i, y + j
                            if 0 <= px < width and 0 <= py < height:
                                weight = kernel[i + radius, j + radius]
                                if weight > 0:
                                    heatmap[py, px] = (weight, weight, weight)

                cv2.imwrite(os.path.join(out_clip_dir, fname), heatmap)


def create_splits(img_dir, out_dir, train_ratio=0.7):
    print("Creating train/val splits...")
    all_labels = pd.DataFrame()

    for game_id in range(1, 11):
        game_dir = f"game{game_id}"
        clips = os.listdir(os.path.join(img_dir, game_dir))
        print(f"  {game_dir}")

        for clip in clips:
            print(f"    {clip}")

            df = pd.read_csv(os.path.join(img_dir, game_dir, clip, "Label.csv"))
            df["gt_path"] = f"gts/{game_dir}/{clip}/" + df["file name"]
            df["image_path"] = f"images/{game_dir}/{clip}/" + df["file name"]

            temporal_df = df[2:].copy()
            temporal_df.loc[:, "prev_image_path"] = list(df["image_path"][1:-1])
            temporal_df.loc[:, "prev2_image_path"] = list(df["image_path"][:-2])

            all_labels = pd.concat([all_labels, temporal_df], ignore_index=True)

    all_labels = all_labels.reset_index(drop=True)
    all_labels = all_labels[
        [
            "image_path",
            "prev_image_path",
            "prev2_image_path",
            "gt_path",
            "x-coordinate",
            "y-coordinate",
            "status",
            "visibility",
        ]
    ]

    all_labels = all_labels.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(all_labels) * train_ratio)

    train_df = all_labels[:n_train]
    val_df = all_labels[n_train:]

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")


def main():
    generate_heatmaps(
        img_dir="data/images",
        out_dir="data/gts",
        radius=20,
        var=10,
        width=1280,
        height=720,
    )

    create_splits(img_dir="data/images", out_dir="data")


if __name__ == "__main__":
    main()
