import logging
from pathlib import Path

import airlab as al
import torch
from monai import transforms as tfm
from monai.data import DataLoader, Dataset, MetaTensor
from monai.transforms import Transform
from tqdm import tqdm


class DirectoryRegistration:
    def __init__(self, directory: Path, image_pattern="CTres_*.nii.gz"):
        self.directory = Path(directory)
        self.input_dir = self.directory / "processed"
        self.output_dir = self.directory / "registered2"
        self.fixed_image_path = self.directory / "mean_image.th"
        self.image_pattern = image_pattern

        self.output_dir.mkdir(exist_ok=True)

        self.data = self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_data(self):
        data = []
        for sample in self.input_dir.glob(self.image_pattern):

            sample_dict = {"img": str(sample)}

            # TODO needs refactoring. We should make one directory per patient!
            extension = sample.name.split("_", 1)[1]
            seg_path = sample.parent / f"SEG_{extension}"

            if seg_path.is_file():
                sample_dict["seg"] = str(seg_path)

            data.append(sample_dict)

        return data

    def get_transform(self):
        with open(self.fixed_image_path, mode="rb") as file:
            mean_image = torch.load(file)

        return tfm.Compose(
            [
                tfm.LoadImaged(keys=["img", "seg"], allow_missing_keys=True),
                RigidTransformd(
                    fixed_image=mean_image, image_key="img", label_key="seg"
                ),
                tfm.SaveImaged(
                    keys=["img", "seg"],
                    output_dir=self.output_dir,
                    resample=False,
                    print_log=False,
                    separate_folder=False,
                    output_postfix="",
                    allow_missing_keys=True,
                ),
            ]
        )

    def run_registration(self, num_workers=8):

        # Solves this isse: https://github.com/pytorch/pytorch/issues/40403
        torch.multiprocessing.set_start_method("spawn")

        dataset = Dataset(self.data, transform=self.get_transform())
        loader = DataLoader(dataset=dataset, num_workers=num_workers)

        self.output_dir.mkdir(exist_ok=True)

        for _ in tqdm(loader, total=len(dataset), desc="Rigistration"):
            continue


def run_rigid_transformation(
    fixed_image: al.Image, moving_image: al.Image, verbose=False
):

    # create pairwise registration object
    registration = al.PairwiseRegistration(verbose=verbose)

    # choose the affine transformation model
    transformation = al.transformation.pairwise.RigidTransformation(
        moving_image, opt_cm=True
    )
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.1)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(50)

    # start the registration
    registration.start()

    return transformation.get_displacement()


class RigidTransformd(Transform):
    def __init__(self, fixed_image: torch.Tensor, image_key="img", label_key="seg"):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.fixed_image = self._parse_to_3d(fixed_image).to(device=self.device)
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data: MetaTensor):

        moving_meta = data[self.image_key].meta
        if self.label_key in data:
            label_meta = data[self.label_key].meta

        moving_image = self._parse_to_3d(data[self.image_key])
        if self.label_key in data:
            label_image = self._parse_to_3d(data[self.label_key])

        moving_image.to(device=self.device)
        if self.label_key in data:
            label_image.to(device=self.device)

        displacement = run_rigid_transformation(self.fixed_image, moving_image)

        warped_image = al.transformation.utils.warp_image(moving_image, displacement)
        if self.label_key in data:
            warped_label = al.transformation.utils.warp_image(label_image, displacement)

        data[self.image_key] = MetaTensor(warped_image.numpy(), meta=moving_meta)
        if self.label_key in data:
            data[self.label_key] = MetaTensor(warped_label.numpy(), meta=label_meta)

        return data

    def _parse_to_3d(self, image):
        if len(image.shape) == 5:
            return al.Image(image[0, 0, ...])
        elif len(image.shape) == 4:
            return al.Image(image[0, ...])
        elif len(image.shape) == 3:
            return al.Image(image)
        else:
            raise ValueError(f"Unsopported fixed image shape {image.shape}")
