from pathlib import Path
from typing import Tuple

import wandb
from meddist.config import init_wandb
from meddist.training import barlow, contrastive, phys
from monai.utils.misc import set_determinism

TRAIN_SCRIPS = {
    "meddist": phys.train,
    "simclr": contrastive.train,
    "barlow": barlow.train,
}

DATA_PATHS = {
    "brats": "/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/data_splits/brats_split.pkl",
    "autopet": "/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/data_splits/autopep_split.pkl",
}


def parse_wandb_config() -> Tuple[str, str, callable]:

    model_log_path = Path(
        f"/sc-scratch/sc-scratch-gbm-radiomics/meddist_models/{wandb.config.model}_{wandb.config.data}/{wandb.run.name}_{wandb.run.id}"
    )

    n_crops = 16384 / (wandb.config.crop_size * wandb.config.batch_size)
    n_crops = max(int(n_crops), 1)

    if n_crops < wandb.config.number_of_crops:
        wandb.config.update({"number_of_crops": n_crops}, allow_val_change=True)

    return model_log_path


if __name__ == "__main__":

    set_determinism()

    # Read config file specifed in command line arguments.
    init_wandb(project_name="Meddist-test")

    model_log_path = parse_wandb_config()
    train_script = TRAIN_SCRIPS[wandb.config.model]
    path_to_data_split = DATA_PATHS[wandb.config.data]

    train_script(path_to_data_split, model_log_path)
