import argparse

import wandb


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        type=str,
        help="Path to config file.",
        default="/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/meddist/configs/simple.yaml",
    )
    return parser.parse_args()


def init_wandb():
    args = read_args()
    wandb.init(project="Meddist", config=args.config)
