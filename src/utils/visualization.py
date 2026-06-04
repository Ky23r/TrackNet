from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import INFERENCE_CONFIG


def draw_trajectory_on_frames(
    frames: List[np.ndarray],
    trajectory: List[Tuple[Optional[float], Optional[float]]],
    trace_length: int = INFERENCE_CONFIG.default_trace_length,
    ball_color: Tuple[int, int, int] = INFERENCE_CONFIG.ball_marker_color,
    ball_radius: int = INFERENCE_CONFIG.ball_marker_radius,
    ball_thickness: int = INFERENCE_CONFIG.ball_marker_thickness,
) -> List[np.ndarray]:
    annotated = []

    for i, frame in enumerate(frames):
        canvas = frame.copy()
        x, y = trajectory[i]

        if x is not None and y is not None:
            cv2.circle(
                canvas, (int(x), int(y)), ball_radius, ball_color, ball_thickness
            )

        for j in range(max(0, i - trace_length), i):
            x0, y0 = trajectory[j]
            x1, y1 = trajectory[j + 1]
            if all(c is not None for c in (x0, y0, x1, y1)):
                cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), ball_color, 2)

        annotated.append(canvas)

    return annotated


def draw_single_frame_annotation(
    frame: np.ndarray,
    position: Tuple[Optional[float], Optional[float]],
    ball_color: Tuple[int, int, int] = INFERENCE_CONFIG.ball_marker_color,
    ball_radius: int = INFERENCE_CONFIG.ball_marker_radius,
    ball_thickness: int = INFERENCE_CONFIG.ball_marker_thickness,
    show_crosshair: bool = False,
) -> np.ndarray:
    canvas = frame.copy()
    x, y = position

    if x is not None and y is not None:
        xi, yi = int(x), int(y)
        cv2.circle(canvas, (xi, yi), ball_radius, ball_color, ball_thickness)

        if show_crosshair:
            arm = 20
            cv2.line(canvas, (xi - arm, yi), (xi + arm, yi), ball_color, 2)
            cv2.line(canvas, (xi, yi - arm), (xi, yi + arm), ball_color, 2)

    return canvas


def create_side_by_side_comparison(
    frame: np.ndarray,
    heatmap: np.ndarray,
    predicted_position: Tuple[Optional[float], Optional[float]],
    ground_truth_position: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    if heatmap.shape[:2] != frame.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    frame_annotated = draw_single_frame_annotation(frame, predicted_position)

    if ground_truth_position is not None:
        x_gt, y_gt = ground_truth_position
        if x_gt > 0 and y_gt > 0:
            cv2.circle(heatmap_colored, (int(x_gt), int(y_gt)), 8, (255, 255, 255), 2)

    return np.hstack([frame_annotated, heatmap_colored])
