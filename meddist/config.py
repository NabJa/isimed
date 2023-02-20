import argparse

import wandb
import yaml

ARGUMENTS = [("data", str), ("model", str), ("crop_size", int), ("temperature", float), ("batch_size", int), ("epochs", int)]


def read_yaml(path):
    with open(path, mode="r") as file:
        return yaml.safe_load(file)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        type=str,
        help="Path to config file.",
        default="/sc-projects/sc-proj-gbm-radiomics/whole-body/meddistssl/configs/train.yaml",
    )
    for arg, arg_type in ARGUMENTS:
        parser.add_argument(f"--{arg}", type=arg_type)

    return parser.parse_args()


def init_wandb(project_name="Meddist"):
    args = read_args()

    run = wandb.init(project=project_name, config=args.config)

    for arg, arg_type in ARGUMENTS:
        argument = args.__getattribute__(arg)
        if argument is not None:
            wandb.config.update({arg: arg_type(argument)}, allow_val_change=True)

    return run
