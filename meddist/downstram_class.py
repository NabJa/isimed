import argparse
import pickle
from pathlib import Path

import monai.transforms as tfm
import numpy as np
import torch
import wandb
from meddist.nets import LinearHead, load_densenet
from meddist.transforms import GetClassesFromCropsd
from monai.data import DataLoader, Dataset
from monai.metrics import ConfusionMatrixMetric
from monai.utils.misc import set_determinism
from torch import nn

torch.multiprocessing.set_sharing_strategy("file_system")
set_determinism()


def get_latest_model(path_to_model_directory):
    directory = Path(path_to_model_directory).iterdir()
    # Model name has the format ARCHITECTURE_EPOCH001.pt
    sorted_models = sorted(directory, key=lambda x: int(x.name.split(".")[0][-3:]))
    return sorted_models[-1]


def get_data_loaders(crop_size=64):
    with open(wandb.config.path_to_data_split, mode="rb") as file:
        split = pickle.load(file)

    transform = tfm.Compose(
        [
            tfm.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            tfm.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                select_fn=lambda x: x > 0,
            ),
            tfm.ScaleIntensityRangePercentilesd(
                keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
            ),
            tfm.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                pos=0.65,
                num_samples=32,
                spatial_size=crop_size,
            ),
            GetClassesFromCropsd(label_key="label"),
        ]
    )

    dataset_valid = Dataset(split["validation"], transform=transform)
    loader_valid = DataLoader(dataset_valid, num_workers=8)

    dataset_train = Dataset(split["test"], transform=transform)
    loader_train = DataLoader(dataset_train, num_workers=8)

    return loader_train, loader_valid


def run_epoch(
    model: nn.Module,
    loss_fn,
    loader,
    optimizer=None,
    metric_names=["sensitivity", "specificity", "accuracy", "f1 score"],
):
    mode = "valid" if optimizer is None else "train"

    confusion = ConfusionMatrixMetric(metric_name=metric_names)

    if mode == "valid":
        model.eval()
    else:
        model.train()

    running_loss = 0.0
    predictions = []

    for i, batch in enumerate(loader):
        image = batch["image"].to("cuda")
        label = batch["has_pos_voxels"].float().unsqueeze(1)

        if mode == "train":
            optimizer.zero_grad()

        pred = model(image).cpu()
        loss = loss_fn(pred, label)

        if mode == "train":
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if mode == "train":
            wandb.log({f"{mode}/Loss": loss.item()})

        pred_binary = (nn.Sigmoid()(pred) > 0.5).int()

        predictions += pred_binary.flatten().tolist()

        confusion(pred_binary, label)

    metrics = confusion.aggregate()

    # metric_dict = dict(zip([f"{mode}/{x}" for x in metric_names], metrics))

    # if mode == "valid":
    #     metric_dict[f"{mode}/Loss"] = running_loss / i

    # wandb.log(metric_dict, commit=False)

    return metrics, running_loss / i


def train(path_to_model_directory):

    metric_names = ["sensitivity", "specificity", "accuracy", "f1 score"]
    path_to_model = get_latest_model(path_to_model_directory)

    classifier = LinearHead(
        load_densenet(path_to_model),
        retrain_backbone=wandb.config.retrain_backbone,
    ).to("cuda")
    optimizer = torch.optim.Adam(classifier.parameters())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(wandb.config.pos_weight))
    )

    loader_train, loader_valid = get_data_loaders(wandb.config.crop_size)

    all_metrics, all_losses, epochs = [], [], []
    for epoch in range(wandb.config.epochs):
        _ = run_epoch(classifier, loss_fn, loader_train, optimizer)
        with torch.no_grad():
            metrics, loss = run_epoch(classifier, loss_fn, loader_valid)
            all_metrics.append(metrics)
            all_losses.append(loss)
            epochs.append(epoch)

    # Log only best epoch
    best_epoch = np.argmin(all_losses)
    log_dict = dict(zip(metric_names, all_metrics[best_epoch]))
    log_dict["Loss"] = all_losses[best_epoch]
    log_dict["Best epoch"] = epochs[best_epoch]

    wandb.log(log_dict, commit=False)


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        type=str,
        help="Path to config file.",
        default="/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/configs/classification.yaml",
    )
    args = parser.parse_args()

    wandb.init(project="Meddist_Class", config=args.config)


if __name__ == "__main__":
    init()
    train()
