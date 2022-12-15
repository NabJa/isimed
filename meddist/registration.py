import airlab as al
import torch
from monai import transforms as tfm
from monai.data import DataLoader
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Transform
from torch.utils.data import DataLoader as dl


def run_rigid_transformation(fixed_image: al.Image, moving_image: al.Image):
    # create pairwise registration object
    registration = al.PairwiseRegistration(verbose=False)

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
    registration.set_number_of_iterations(500)

    # start the registration
    registration.start()

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)

    return warped_image, displacement


class RigidTransformd(Transform):
    def __init__(self, fixed_image: torch.Tensor, image_key="image"):
        self.fixed_image = al.Image(fixed_image[0]) # Remove channel dim
        self.image_key = image_key

    def __call__(self, data: MetaTensor):
        moving_image = al.Image(data[self.image_key][0]) # Remove channel dim

        warped_image, _ = run_rigid_transformation(self.fixed_image, moving_image)

        data[self.image_key] = warped_image

        return data
