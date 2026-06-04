from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import distance as scipy_distance

from ..config import EVALUATION_CONFIG, INFERENCE_CONFIG, MODEL_CONFIG


def extract_ball_position(
    heatmap: np.ndarray,
    scale: int = INFERENCE_CONFIG.heatmap_scale_factor,
) -> Tuple[Optional[float], Optional[float]]:
    if heatmap.ndim == 1:
        heatmap = heatmap.reshape(MODEL_CONFIG.input_height, MODEL_CONFIG.input_width)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    _, binary = cv2.threshold(
        heatmap_uint8, INFERENCE_CONFIG.heatmap_threshold, 255, cv2.THRESH_BINARY
    )

    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=INFERENCE_CONFIG.hough_dp,
        minDist=INFERENCE_CONFIG.hough_min_dist,
        param1=INFERENCE_CONFIG.hough_param1,
        param2=INFERENCE_CONFIG.hough_param2,
        minRadius=INFERENCE_CONFIG.hough_min_radius,
        maxRadius=INFERENCE_CONFIG.hough_max_radius,
    )

    if circles is not None and len(circles) == 1:
        return float(circles[0][0][0] * scale), float(circles[0][0][1] * scale)

    return None, None


def calculate_metrics(
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
    epsilon: float = EVALUATION_CONFIG.epsilon,
) -> Tuple[float, float, float]:
    precision = true_positives / (true_positives + false_positives + epsilon)
    total_positive = true_positives + false_positives + false_negatives
    recall = true_positives / (total_positive + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, f1


class ConfusionMatrix:
    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.tn = [0] * self.num_classes
        self.fn = [0] * self.num_classes

    def update(
        self,
        predicted_position: Tuple[Optional[float], Optional[float]],
        ground_truth_x: float,
        ground_truth_y: float,
        visibility: int,
        distance_threshold: float = 5.0,
    ) -> None:
        vis = int(visibility)
        if not (0 <= vis < self.num_classes):
            return

        x_pred, y_pred = predicted_position
        ball_detected = x_pred is not None
        ball_visible = vis != 0

        if ball_detected:
            if ball_visible:
                dist = scipy_distance.euclidean(
                    (x_pred, y_pred), (ground_truth_x, ground_truth_y)
                )
                if dist < distance_threshold:
                    self.tp[vis] += 1
                else:
                    self.fp[vis] += 1
            else:
                self.fp[vis] += 1
        else:
            if ball_visible:
                self.fn[vis] += 1
            else:
                self.tn[vis] += 1

    def get_totals(self) -> Tuple[int, int, int, int]:
        return sum(self.tp), sum(self.fp), sum(self.tn), sum(self.fn)

    def compute_metrics(self) -> Tuple[float, float, float]:
        tp, fp, tn, fn = self.get_totals()
        return calculate_metrics(tp, fp, tn, fn)

    def __str__(self) -> str:
        tp, fp, tn, fn = self.get_totals()
        precision, recall, f1 = self.compute_metrics()
        return (
            f"ConfusionMatrix(\n"
            f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}\n"
            f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n"
            f")"
        )
