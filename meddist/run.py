import wandb
from meddist.config import init_wandb
from meddist.training import barlow, contrastive, phys
from monai.utils.misc import set_determinism

if __name__ == "__main__":

    set_determinism()

    # Read config file specifed in command line arguments.
    init_wandb()

    if wandb.config.data == "brats":
        path_to_data_split = "/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/data_splits/brats_split.pkl"
    elif wandb.config.data == "autopep":
        path_to_data_split = "/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/data_splits/autopep_split.pkl"
    else:
        raise NotImplementedError(
            f"Dataset ' {wandb.config.data}' unknown. Choose between brats and autopet."
        )

    # New training run
    if wandb.config.model == "meddist":
        phys.train(path_to_data_split)
    elif wandb.config.model == "simclr":
        contrastive.train(path_to_data_split)
    elif wandb.config.model == "barlow":
        barlow.train(path_to_data_split)
    else:
        raise NotImplementedError(
            f"Model '{wandb.config.model}' unknown. Choose between meddist, simclr and barlow."
        )
