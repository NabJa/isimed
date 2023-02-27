import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from monai.utils.misc import set_determinism
from torch import nn
from tqdm import tqdm, trange

from meddist.data.loading import kfold_get_downstram_data
from meddist.metrics import ClassificationMetricTracker, RegressionMetricTracker
from meddist.nets import LinearHead, load_latest_densenet

torch.multiprocessing.set_sharing_strategy("file_system")
set_determinism()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

        # Forward
        pred = model(image).cpu()
        loss = loss_fn(pred, label)

        # Backward
        if train_mode:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Loss
        running_loss += loss.item()

        # Confusion matrix
        if metric_tracker is not None:
            with torch.no_grad():
                metric_tracker(pred.detach().as_tensor(), label)

    # Aggregate all
    metrics = metric_tracker.aggregate() if metric_tracker is not None else {}
    running_loss = running_loss / (i + 1)

    return metrics, running_loss


def run_downstream(
    train_loader,
    valid_loader,
    model,
    epochs=50,
    task="classification",
) -> Tuple[List[dict], List[float]]:
    """
    One downstream run. This function is run K times in a KFold CV.

    Args:
        path_to_data_split: Path to data split file.
        path_to_model_dir: Path to model directory containing trained models.
        task: Classification or Regression. Defaults to "classification".
    """

    loss_fn, metric_tracker, label_key = TASK_PREP[task.lower()]

    optimizer = torch.optim.Adam(model.parameters())

    all_valid_metrics = []
    all_valid_losses = []
    for epoch in trange(epochs, desc="Epoch"):
        _, _ = run_epoch(model, loss_fn, train_loader, label_key, optimizer=optimizer)
        with torch.no_grad():
            valid_metrics, valid_loss = run_epoch(
                model, loss_fn, valid_loader, label_key, metric_tracker=metric_tracker
            )

        all_valid_metrics.append(valid_metrics)
        all_valid_losses.append(valid_loss)

    return all_valid_metrics, all_valid_losses


def run_kfold_downstream_experiment(
    path_to_data_split,
    path_to_model_dir,
    kfolds=5,
    retrain_backbone=False,
    crop_size=32,
    batch_size=32,
    epochs=10,
    task="classification",
    num_workers=8,
):
    """Iteration of KFold Cross Validation on Valid data."""

    data_loader_generator = kfold_get_downstram_data(
        path_to_data_split, kfolds, crop_size, batch_size, num_workers, task
    )

    metrics, losses = [], []
    for fold, (train_loader, valid_loader) in enumerate(data_loader_generator):

        model = LinearHead(
            load_latest_densenet(path_to_model_dir),
            retrain_backbone=retrain_backbone,
            final_activation="RELU" if task == "regression" else None,
        ).to(DEVICE)

        metric, loss = run_downstream(
            train_loader, valid_loader, model, epochs, task=task
        )
        metrics.append(metric)
        losses.append(loss)

    return metrics, losses


def kfold_results_to_df(results):
    """Utility function to transform results to dataframe."""
    df = []
    for fold, (model, (metrics, losses)) in enumerate(results.items()):
        for metric, loss in zip(metrics, losses):
            for metric_name, value in metric[np.argmin(loss)].items():
                df.append((model, fold, metric_name, value))

    return pd.DataFrame(df, columns=["Model", "BestFold", "Metric", "Value"])


def kfold_on_all_models(
    split_path: str, model_paths: dict, **experiment_kwags
) -> pd.DataFrame:
    """Main function for KFold Cross Validation experiment on all models."""
    results = {}
    for name, model_path in tqdm(model_paths.items()):
        metrics, losses = run_kfold_downstream_experiment(
            split_path, model_path, **experiment_kwags
        )
        results[name] = (metrics, losses)

    return kfold_results_to_df(results)


if __name__ == "__main__":

    path_to_autopet_models = {
        "Meddist": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/meddist_autopet/likely-sweep-3_4u7nq7wr",
        "simCLR": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/simclr_autopet/revived-sweep-2_bv39ht3n",
        "Barlow": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/barlow_autopet/rare-sweep-1_wm8xqipy",
        "BarlowDist": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/barlowdist_autopet/misunderstood-sweep-6_ffd48pdy",
        "Hydra": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/hydra_autopet/volcanic-sweep-6_9fckn0b2",
    }

    path_to_brats_models = {
        "Meddist": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/meddist_brats/woven-sweep-11_3mtl3e2q",
        "simCLR": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/simclr_brats/daily-sweep-10_139fn4em",
        "Barlow": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/barlow_brats/jolly-sweep-9_kkcwem37",
        "BarlowDist": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/barlowdist_brats/skilled-sweep-5_yyajynx8",
        "Hydra": "/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/hydra_brats/chocolate-sweep-5_xjpbcdxw",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str)
    parser.add_argument("-task", type=str)
    parser.add_argument("-out_file", type=str)
    parser.add_argument("-folds", type=int)
    parser.add_argument("-epochs", default=25, type=int)
    parser.add_argument("-crop_size", type=int, default=32)
    parser.add_argument("-num_workers", type=int, default=24)
    args = parser.parse_args()

    if args.data == "autopet":
        split_path = "/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/data_splits/autopep_split.pkl"
        model_paths = path_to_autopet_models
    elif args.data == "brats":
        split_path = "/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/data_splits/brats_split.pkl"
        model_paths = path_to_brats_models

    df = kfold_on_all_models(
        split_path,
        model_paths,
        kfolds=args.folds,
        task=args.task,
        epochs=args.epochs,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
    )
    df.to_csv(args.out_file)
