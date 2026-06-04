from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance as scipy_distance
from tqdm import tqdm

from ..config import EVALUATION_CONFIG
from ..utils.metrics import ConfusionMatrix, extract_ball_position


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        distance_threshold: int = EVALUATION_CONFIG.distance_threshold_pixels,
    ) -> None:
        self.model = model
        self.device = device
        self.distance_threshold = distance_threshold
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.model.eval()

        losses = []
        confusion_matrix = ConfusionMatrix(num_classes=4)

        desc = f"Val {epoch}" if epoch is not None else "Evaluation"
        pbar = tqdm(data_loader, desc=desc)

        for batch in pbar:
            batch_result = self._evaluate_batch(batch, confusion_matrix)
            losses.append(batch_result["loss"])
            tp, fp, tn, fn = confusion_matrix.get_totals()
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}", tp=tp, tn=tn, fp=fp, fn=fn)

        avg_loss = float(np.mean(losses))
        precision, recall, f1_score = confusion_matrix.compute_metrics()
        tp, fp, tn, fn = confusion_matrix.get_totals()

        return {
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

    def _evaluate_batch(
        self,
        batch: Any,
        confusion_matrix: ConfusionMatrix,
    ) -> Dict[str, float]:
        with torch.no_grad():
            frames = batch[0].float().to(self.device)
            target_heatmap = batch[1].long().to(self.device)

            output = self.model(frames, inference=False)
            loss = self.criterion(output, target_heatmap)
            predictions = output.argmax(dim=1).cpu().numpy()

            for i in range(len(predictions)):
                x_pred, y_pred = extract_ball_position(predictions[i])
                confusion_matrix.update(
                    predicted_position=(x_pred, y_pred),
                    ground_truth_x=batch[2][i].item(),
                    ground_truth_y=batch[3][i].item(),
                    visibility=batch[4][i].item(),
                    distance_threshold=self.distance_threshold,
                )

        return {"loss": loss.item()}

    def evaluate_single_sample(
        self,
        frames: torch.Tensor,
        ground_truth_x: float,
        ground_truth_y: float,
        visibility: int,
    ) -> Dict[str, Any]:
        self.model.eval()

        with torch.no_grad():
            frames = frames.float().to(self.device)
            output = self.model(frames, inference=True)
            prediction = output.argmax(dim=1).cpu().numpy()[0]
            x_pred, y_pred = extract_ball_position(prediction)

            ball_detected = x_pred is not None
            ball_visible = visibility != 0

            result: Dict[str, Any] = {
                "predicted_x": x_pred,
                "predicted_y": y_pred,
                "ground_truth_x": ground_truth_x,
                "ground_truth_y": ground_truth_y,
                "visibility": visibility,
                "ball_detected": ball_detected,
                "ball_visible": ball_visible,
            }

            if ball_detected and ball_visible:
                distance = scipy_distance.euclidean(
                    (x_pred, y_pred), (ground_truth_x, ground_truth_y)
                )
                result["distance"] = distance
                result["correct"] = distance < self.distance_threshold
            else:
                result["distance"] = None
                result["correct"] = ball_detected == ball_visible

        return result

    def compute_class_wise_metrics(
        self,
        data_loader: torch.utils.data.DataLoader,
    ) -> Dict[int, Dict[str, Any]]:
        self.model.eval()

        class_confusion = {i: ConfusionMatrix(num_classes=4) for i in range(4)}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing class-wise metrics"):
                frames = batch[0].float().to(self.device)
                output = self.model(frames, inference=False)
                predictions = output.argmax(dim=1).cpu().numpy()

                for i in range(len(predictions)):
                    x_pred, y_pred = extract_ball_position(predictions[i])
                    visibility = batch[4][i].item()

                    if 0 <= visibility < 4:
                        class_confusion[visibility].update(
                            (x_pred, y_pred),
                            batch[2][i].item(),
                            batch[3][i].item(),
                            visibility,
                            self.distance_threshold,
                        )

        results: Dict[int, Dict[str, Any]] = {}
        for vis_class, cm in class_confusion.items():
            precision, recall, f1 = cm.compute_metrics()
            tp, fp, tn, fn = cm.get_totals()
            results[vis_class] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "total_samples": tp + fp + tn + fn,
            }

        return results
