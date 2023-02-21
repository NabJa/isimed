import os
from pathlib import Path
from typing import Tuple

from monai.utils.misc import set_determinism

import wandb
from meddist.config import init_wandb
from meddist.training.train import train

DATA_PATHS = {
    "brats": "/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/data_splits/brats_split.pkl",
    "autopet": "/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/data_splits/autopep_split.pkl",
}

MODEL_LOG_ROOT = Path("/sc-scratch/sc-scratch-gbm-radiomics/meddist_models")


def cleanup_model_log_directory(root):
    """Delete all empty sub-direcoties from root."""
    walk = list(os.walk(root))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


def get_ncrops(
    batch_size, crop_size, num_allowed_voxels=50_000_000, min_crops=1, max_crops=512
) -> int:
    """Compute the number of crops for a given batch_size and crop_size.
    Returns the number of crops so that ncrops * batch_size * crop_size**3 < num_allowed_voxels."""
    ncrops = num_allowed_voxels / (batch_size * crop_size ** 3)
    ncrops = int((ncrops // 8) * 8)  # Make ncrops divisable by 8
    return min(max(ncrops, min_crops), max_crops)


def parse_wandb_config() -> Tuple[str, str, callable]:

    model_log_path = (
        MODEL_LOG_ROOT
        / f"{wandb.config.model}_{wandb.config.data}/{wandb.run.name}_{wandb.run.id}"
    )

    ncrops = get_ncrops(wandb.config.batch_size, wandb.config.crop_size)
    wandb.config.update({"number_of_crops": ncrops}, allow_val_change=True)

    return model_log_path


if __name__ == "__main__":

    set_determinism()

    with init_wandb(project_name="SLLMedImages"):
        model_log_path = parse_wandb_config()
        path_to_data_split = DATA_PATHS[wandb.config.data]
        train(path_to_data_split, model_log_path)

    # cleanup_model_log_directory(MODEL_LOG_ROOT)
