from typing import Tuple

import numpy as np
import torch
from monai.metrics import ConfusionMatrixMetric, MAEMetric, RMSEMetric
from monai.utils.misc import set_determinism
from torch import nn

import wandb
from meddist.data.loading import get_downstram_classification_data
from meddist.nets import LinearHead, load_latest_densenet

torch.multiprocessing.set_sharing_strategy("file_system")
set_determinism()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RegressionMetricTracker:
    def __init__(self) -> None:
        self.rmse = RMSEMetric()
        self.mae = MAEMetric()

    def __call__(self, y_pred, y) -> None:
        self.rmse(y_pred, y)
        self.mae(y_pred, y)

    def aggregate(self):

        if len(self.rmse) == 0 or len(self.mae) == 0:
            return {"RMSE": np.inf, "MAE": np.inf}

        rmse = self.rmse.aggregate()
        mae = self.mae.aggregate()
        return {"RMSE": rmse.item(), "MAE": mae.item()}

    def reset(self):
        self.rmse.reset()
        self.mae.reset()


class ClassificationMetricTracker:
    def __init__(
        self,
        threshold=0.5,
        metric_names=("sensitivity", "specificity", "accuracy", "f1 score"),
    ) -> None:
        self.threshold = threshold
        self.metric_names = metric_names
        self.confusion = ConfusionMatrixMetric(metric_name=metric_names)
        self.sigmoid = nn.Sigmoid()

    def __call__(self, y_pred, y) -> None:
        y_pred = (self.sigmoid(y_pred) > self.threshold).int()
        self.confusion(y_pred, y)

    def aggregate(self):
        values = self.confusion.aggregate()
        return {name: value.item() for name, value in zip(self.metric_names, values)}

    def reset(self):
        self.confusion.reset()


TASK_PREP = {
    "classification": (
        nn.BCEWithLogitsLoss(),
        ClassificationMetricTracker(),
        "has_pos_voxels",
    ),
    "regression": (nn.MSELoss(), RegressionMetricTracker(), "num_pos_voxels"),
}


def run_epoch(
    model, loss_fn, loader, label_key, metric_tracker=None, optimizer=None
) -> Tuple[dict, float]:

    train_mode = optimizer is not None

    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    for i, batch in enumerate(loader):
        image = batch["image"].to(DEVICE)
        label = batch[label_key].float().unsqueeze(1)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            pred = model(image).cpu()

        loss = loss_fn(pred, label)

        if train_mode:
            loss.backward()
            optimizer.step()

        # Loss
        running_loss += loss.item()

        # Confusion matrix
        if metric_tracker is not None:
            metric_tracker(pred, label)

    # Aggregate all
    metrics = metric_tracker.aggregate() if metric_tracker is not None else {}
    running_loss = running_loss / i

    return metrics, running_loss


def run_downstream(
    path_to_data_split,
    path_to_model_dir,
    retrain_backbone=False,
    crop_size=32,
    epochs=50,
    task="classification",
):
    """
    Args:
        path_to_data_split: Path to data split file.
        path_to_model_dir: Path to model directory containing trained models.
        task: Classification or Regression. Defaults to "classification".
    """

    loss_fn, metric_tracker, label_key = TASK_PREP[task.lower()]

    model = LinearHead(
        load_latest_densenet(path_to_model_dir),
        retrain_backbone=retrain_backbone,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    train_loader, valid_loader = get_downstram_classification_data(
        path_to_data_split, crop_size
    )

    for epoch in range(epochs):
        metrics, loss = run_epoch(
            model, loss_fn, train_loader, label_key, optimizer=optimizer
        )
        print("Train loss", loss)
        print("Train metrics", metrics)
        with torch.no_grad():
            metrics, loss = run_epoch(
                model, loss_fn, valid_loader, label_key, metric_tracker=metric_tracker
            )
            print("Valid loss", loss)
            print("Valid metrics", metrics)
