from abc import abstractmethod

import torch
from monai.metrics import (
    compute_confusion_matrix_metric,
    compute_roc_auc,
    get_confusion_matrix,
)
from torch import nn


class Cumulative:
    def __init__(self) -> None:
        self.reset()

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """Add data to buffers. Data is flattened and ignores batches. Must be same shape."""

        assert (
            y_pred.shape == y.shape
        ), f"Given shapes: y_pred={y_pred.shape} y={y.shape}"

        self.pred_buffer.append(y_pred)
        self.label_buffer.append(y)

    def reset(self):
        """Init/Empty buffers"""
        self.pred_buffer = []
        self.label_buffer = []

    @abstractmethod
    def aggregate(self, *args, **kwargs):
        """
        Aggregate final results based on the gathered buffers.
        Computation of metrics based on labels and prediction must be done in this method.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )


class RegressionMetricTracker(Cumulative):
    def __init__(self) -> None:
        super().__init__()

    def aggregate(self):

        preds = torch.cat(self.pred_buffer, dim=0)
        labels = torch.cat(self.label_buffer, dim=0)

        rmse = torch.sqrt(torch.pow(preds - labels, 2).mean()).item()
        mae = torch.abs(preds - labels).mean().item()

        return {"RMSE": rmse, "MAE": mae}


class ClassificationMetricTracker(Cumulative):
    def __init__(
        self,
        threshold=0.5,
        metric_names=("sensitivity", "specificity", "accuracy", "f1 score"),
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.metric_names = metric_names
        self.sigmoid = nn.Sigmoid()

    def aggregate(self):

        preds = torch.cat(self.pred_buffer, dim=0)
        labels = torch.cat(self.label_buffer, dim=0)

        result = {
            "AUC": compute_roc_auc(
                preds.flatten(), labels.flatten()
            )
        }

        pred_binary = (self.sigmoid(preds) > self.threshold).float()
        cm = get_confusion_matrix(pred_binary, labels)
        cm = torch.sum(cm, 0)[0]  # Sum over all batches.

        for name, value in zip(["TP", "FP", "TN", "FN"], cm):
            result[name] = value.item()

        for name in self.metric_names:
            result[name] = compute_confusion_matrix_metric(name, cm).item()

        return result
