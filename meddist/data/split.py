import argparse
import pickle
from pathlib import Path

from meddist.data.autopet import split_autopet_data
from meddist.data.brats import split_brats_data


def save_datasplit(train_data, val_data, test_data, save_to_path, name="split") -> None:
    save_to_path = Path(save_to_path)

    if save_to_path.is_dir():
        save_to_path = save_to_path / f"{name}.pkl"

    with open(save_to_path, mode="wb") as file:
        split = {"train": train_data, "validation": val_data, "test": test_data}
        pickle.dump(split, file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str)
    parser.add_argument("-data_path", type=Path)
    parser.add_argument("-output_path", type=Path)
    parser.add_argument("-valid_size", default=0.2, type=float)
    parser.add_argument("-test_size", default=0.2, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "brats":
        train, valid, test = split_brats_data(
            args.data_path, valid_size=args.valid_size
        )
    elif args.dataset == "autopep":
        train, valid, test = split_autopet_data(
            args.data_path, valid_size=args.valid_size
        )
    else:
        raise NotImplementedError("Only brats and autopep implemented.")

    save_datasplit(train, valid, test, args.output_path, name=f"{args.dataset}_split")
