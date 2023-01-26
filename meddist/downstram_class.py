import argparse
import pickle

import monai.transforms as tfm
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.metrics import ConfusionMatrixMetric
from monai.utils.misc import set_determinism
from torch import nn

import wandb
from meddist.nets import ClassificationHead, load_densenet
from meddist.transforms import GetClassesFromCropsd, RandCropBlanacedd

torch.multiprocessing.set_sharing_strategy("file_system")
set_determinism()


def get_data_loaders():
    with open(wandb.config.path_to_data_split, mode="rb") as file:
        split = pickle.load(file)

    transform = tfm.Compose(
        [
            tfm.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            tfm.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                select_fn=lambda x: x > -1000,
            ),
            tfm.ScaleIntensityRangePercentilesd(
                keys="image", lower=5, upper=95, b_min=-1.0, b_max=1.0
            ),
            RandCropBlanacedd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=128,
                num_samples=5,
            ),
            GetClassesFromCropsd(label_key="label"),
        ]
    )

    dataset_valid = Dataset(split["validation"], transform=transform)
    loader_valid = DataLoader(dataset_valid, num_workers=8)

    dataset_train = Dataset(split["test"], transform=transform)
    loader_train = DataLoader(dataset_train, num_workers=8)

    return loader_train, loader_valid


def iteration(model: nn.Module, loss_fn, loader, optimizer=None):
    mode = "valid" if optimizer is None else "train"

    metric_names = ["sensitivity", "specificity", "accuracy", "f1 score"]
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

        pred_binary = nn.Sigmoid()(pred > 0.5).int()

        predictions += pred_binary.flatten().tolist()

        confusion(pred_binary, label)

    metrics = confusion.aggregate()
    metric_dict = dict(zip([f"{mode}/{x}" for x in metric_names], metrics))

    if mode == "valid":
        metric_dict[f"{mode}/Loss"] = running_loss / i

    metric_dict[f"{mode}/pred_balance"] = np.mean(predictions)

    wandb.log(metric_dict, commit=False)


def train():

    classifier = ClassificationHead(load_densenet(wandb.config.path_to_model)).to(
        "cuda"
    )
    optimizer = torch.optim.Adam(classifier.parameters())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(wandb.config.pos_weight))
    )

    loader_train, loader_valid = get_data_loaders()

    for _ in range(wandb.config.epochs):
        iteration(classifier, loss_fn, loader_train, optimizer)
        with torch.no_grad():
            iteration(classifier, loss_fn, loader_valid)


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        type=str,
        help="Path to config file.",
        default="/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/meddist/configs/classification.yaml",
    )
    args = parser.parse_args()

    wandb.init(project="Meddist_Class", config=args.config)


if __name__ == "__main__":
    init()
    train()
