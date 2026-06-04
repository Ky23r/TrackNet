from itertools import groupby
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial import distance as scipy_distance
from tqdm import tqdm

from ..config import INFERENCE_CONFIG, MODEL_CONFIG
from ..utils.metrics import extract_ball_position
from ..utils.visualization import draw_trajectory_on_frames

Position = Tuple[Optional[float], Optional[float]]


class VideoInferencer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        input_width: int = MODEL_CONFIG.input_width,
        input_height: int = MODEL_CONFIG.input_height,
    ) -> None:
        self.model = model
        self.device = device
        self.input_width = input_width
        self.input_height = input_height
        self.model.eval()

    def infer_on_video(
        self,
        video_path: str,
        output_path: str,
        enable_interpolation: bool = False,
        trace_length: int = INFERENCE_CONFIG.default_trace_length,
    ) -> Dict[str, object]:
        print(f"Loading video: {video_path}")
        frames, fps = self._load_video(video_path)
        original_height, original_width = frames[0].shape[:2]
        print(
            f"Video: {len(frames)} frames, {fps} FPS, {original_width}x{original_height}"
        )

        print("Running inference...")
        trajectory, distances = self._predict_trajectory(
            frames, original_width, original_height
        )

        print("Post-processing trajectory...")
        trajectory = self._remove_outliers(trajectory, distances)

        if enable_interpolation:
            print("Interpolating trajectory gaps...")
            trajectory = self._interpolate_trajectory(trajectory)

        print(f"Generating output video: {output_path}")
        self._save_video(frames, trajectory, output_path, fps, trace_length)

        detected = sum(1 for pos in trajectory if pos[0] is not None)
        rate = detected / len(trajectory)

        print(f"Detection rate: {rate:.2%} ({detected}/{len(trajectory)} frames)")

        return {
            "total_frames": len(frames),
            "detected_frames": detected,
            "detection_rate": rate,
            "output_path": output_path,
            "fps": fps,
            "resolution": (original_width, original_height),
        }

    def _load_video(self, video_path: str) -> Tuple[List[np.ndarray], int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames: List[np.ndarray] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break

        cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from video: {video_path}")

        return frames, fps

    def _predict_trajectory(
        self,
        frames: List[np.ndarray],
        original_width: int,
        original_height: int,
    ) -> Tuple[List[Position], List[float]]:
        scale_x = original_width / (self.input_width * 2)
        scale_y = original_height / (self.input_height * 2)

        trajectory: List[Position] = [(None, None), (None, None)]
        distances: List[float] = [-1.0, -1.0]

        for i in tqdm(range(2, len(frames)), desc="Predicting"):
            current = cv2.resize(frames[i], (self.input_width, self.input_height))
            prev = cv2.resize(frames[i - 1], (self.input_width, self.input_height))
            prev2 = cv2.resize(frames[i - 2], (self.input_width, self.input_height))

            tensor = self._prepare_input(current, prev, prev2)

            with torch.no_grad():
                output = self.model(tensor, inference=True)

            prediction = output.argmax(dim=1).cpu().numpy()[0]
            x, y = extract_ball_position(prediction)

            if x is not None and y is not None:
                x = x * scale_x
                y = y * scale_y

            trajectory.append((x, y))

            prev_pos = trajectory[i - 1]
            if x is not None and prev_pos[0] is not None:
                distances.append(scipy_distance.euclidean((x, y), prev_pos))
            else:
                distances.append(-1.0)

        return trajectory, distances

    def _prepare_input(
        self,
        current: np.ndarray,
        prev: np.ndarray,
        prev2: np.ndarray,
    ) -> torch.Tensor:
        stacked = (
            np.concatenate((current, prev, prev2), axis=2).astype(np.float32) / 255.0
        )
        stacked = np.moveaxis(stacked, 2, 0)
        return torch.from_numpy(stacked[np.newaxis]).float().to(self.device)

    def _remove_outliers(
        self,
        trajectory: List[Position],
        distances: List[float],
        max_distance: int = INFERENCE_CONFIG.max_outlier_distance,
    ) -> List[Position]:
        result = list(trajectory)
        for idx in np.where(np.array(distances) > max_distance)[0]:
            result[idx] = (None, None)
        return result

    def _interpolate_trajectory(self, trajectory: List[Position]) -> List[Position]:
        result = list(trajectory)
        for start, end in self._segment_trajectory(trajectory):
            segment = trajectory[start:end]
            result[start:end] = self._interpolate_segment(segment)
        return result

    def _segment_trajectory(
        self,
        trajectory: List[Position],
        max_gap: int = INFERENCE_CONFIG.max_trajectory_gap,
        max_gap_distance: int = INFERENCE_CONFIG.max_gap_distance,
        min_length: int = INFERENCE_CONFIG.min_trajectory_length,
    ) -> List[Tuple[int, int]]:
        flags = [0 if pos[0] is not None else 1 for pos in trajectory]
        groups = [(key, sum(1 for _ in grp)) for key, grp in groupby(flags)]

        subtracks: List[Tuple[int, int]] = []
        pos = 0
        start = 0

        for i, (is_gap, length) in enumerate(groups):
            if is_gap == 1:
                if 0 < length <= max_gap and 0 < i < len(groups) - 1:
                    if pos > 0 and pos + length < len(trajectory):
                        before = trajectory[pos - 1]
                        after = trajectory[pos + length]
                        if before[0] is not None and after[0] is not None:
                            if (
                                scipy_distance.euclidean(before, after)
                                <= max_gap_distance
                            ):
                                pos += length
                                continue

                if pos - start >= min_length:
                    subtracks.append((start, pos))
                pos += length
                start = pos
            else:
                pos += length

        if len(trajectory) - start >= min_length:
            subtracks.append((start, len(trajectory)))

        return subtracks

    def _interpolate_segment(self, segment: List[Position]) -> List[Position]:
        x_vals = np.array([p[0] if p[0] is not None else np.nan for p in segment])
        y_vals = np.array([p[1] if p[1] is not None else np.nan for p in segment])

        valid = np.where(~np.isnan(x_vals))[0]
        if len(valid) < 2:
            return segment

        all_idx = np.arange(len(x_vals))
        missing = np.isnan(x_vals)
        x_vals[missing] = np.interp(all_idx[missing], valid, x_vals[valid])
        y_vals[missing] = np.interp(all_idx[missing], valid, y_vals[valid])

        return [(float(x), float(y)) for x, y in zip(x_vals, y_vals)]

    def _save_video(
        self,
        frames: List[np.ndarray],
        trajectory: List[Position],
        output_path: str,
        fps: int,
        trace_length: int,
    ) -> None:
        annotated = draw_trajectory_on_frames(
            frames, trajectory, trace_length=trace_length
        )
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*INFERENCE_CONFIG.video_codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in annotated:
            writer.write(frame)
        writer.release()
