from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    input_channels: int = 9
    output_channels: int = 256
    input_height: int = 360
    input_width: int = 640
    weight_init_min: float = -0.05
    weight_init_max: float = 0.05


@dataclass
class TrainingConfig:
    default_learning_rate: float = 1.0
    default_batch_size: int = 2
    default_epochs: int = 200
    default_max_steps_per_epoch: int = 200
    default_validation_interval: int = 5
    default_num_workers: int = 1


@dataclass
class EvaluationConfig:
    distance_threshold_pixels: int = 5
    epsilon: float = 1e-15


@dataclass
class PreprocessingConfig:
    gaussian_radius: int = 20
    gaussian_variance: int = 10
    original_image_width: int = 1280
    original_image_height: int = 720
    model_input_width: int = 640
    model_input_height: int = 360
    train_ratio: float = 0.7
    random_seed: int = 42
    default_data_root: str = "data"
    images_subdir: str = "images"
    heatmaps_subdir: str = "gts"
    game_id_start: int = 1
    game_id_end: int = 11


@dataclass
class InferenceConfig:
    heatmap_threshold: int = 127
    hough_dp: int = 1
    hough_min_dist: int = 1
    hough_param1: int = 50
    hough_param2: int = 2
    hough_min_radius: int = 2
    hough_max_radius: int = 7
    heatmap_scale_factor: int = 2
    max_outlier_distance: int = 100
    max_trajectory_gap: int = 4
    max_gap_distance: int = 80
    min_trajectory_length: int = 5
    default_trace_length: int = 7
    ball_marker_radius: int = 5
    ball_marker_color: Tuple[int, int, int] = (0, 255, 255)
    ball_marker_thickness: int = -1
    video_codec: str = "mp4v"


@dataclass
class PathConfig:
    experiments_dir: str = "exps"
    pretrained_dir: str = "pretrained"
    input_videos_dir: str = "input_videos"
    output_videos_dir: str = "output_videos"
    best_model_filename: str = "best.pt"
    last_model_filename: str = "last.pt"
    train_csv_filename: str = "train.csv"
    val_csv_filename: str = "val.csv"
    label_csv_filename: str = "Label.csv"


MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
EVALUATION_CONFIG = EvaluationConfig()
PREPROCESSING_CONFIG = PreprocessingConfig()
INFERENCE_CONFIG = InferenceConfig()
PATH_CONFIG = PathConfig()
