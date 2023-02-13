import os
from pathlib import Path
from typing import Tuple

import wandb
from meddist.config import init_wandb
from meddist.training.train import train
from monai.utils.misc import set_determinism

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


def parse_wandb_config() -> Tuple[str, str, callable]:

    model_log_path = MODEL_LOG_ROOT / f"{wandb.config.model}_{wandb.config.data}/{wandb.run.name}_{wandb.run.id}"

    n_crops = 16384 / (wandb.config.crop_size * wandb.config.batch_size)
    n_crops = max(int(n_crops), 1)

    if n_crops < wandb.config.number_of_crops:
        wandb.config.update({"number_of_crops": n_crops}, allow_val_change=True)

    return model_log_path


if __name__ == "__main__":

    set_determinism()

    with init_wandb(project_name="Meddist-test"):
        model_log_path = parse_wandb_config()
        path_to_data_split = DATA_PATHS[wandb.config.data]
        train(path_to_data_split, model_log_path)

    cleanup_model_log_directory(MODEL_LOG_ROOT)
