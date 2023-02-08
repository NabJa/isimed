import wandb
from meddist.config import init_wandb
from meddist.training import contrastive, phys
from monai.utils.misc import set_determinism

if __name__ == "__main__":

    set_determinism()
    
    # Read config file specifed in command line arguments.
    init_wandb()

    # New training run
    if wandb.config.model == "meddist":
        phys.train()
    elif wandb.config.model == "simclr":
        contrastive.train()
