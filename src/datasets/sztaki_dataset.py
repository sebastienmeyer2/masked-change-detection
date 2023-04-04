"""Load and prepare SZTAKI dataset for reconstruction."""


import os

import numpy as np
from PIL import Image

from skimage.exposure import match_histograms

from torch.utils.data import Dataset


from datasets.transforms import create_reconstruction_transform
from utils.image_processing import simple_equalization_8bit, crop_images
from utils.misc import raise_if_no_dir, raise_if_no_file


class SztakiReconstructionDataset(Dataset):
    """Dataset used for reconstruction with OmniMAE."""

    def __init__(
        self, data_dir="data", dataset_name="sztaki", train=True, transform=None,
        percentiles=2, cropping_method="random", img_height=224, img_width=224, left=0, top=0,
        right=1, bottom=1
    ):

        # Parameters
        self.train = train
        self.test_names = ["Szada1", "Tiszadob3"]

        self.transform = transform
        if self.transform is None:
            self.transform = create_reconstruction_transform()

        self.percentiles = percentiles

        self.cropping_method = cropping_method
        self.img_height = img_height
        self.img_width = img_width
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

        # Check that the data directory exists
        raise_if_no_dir(data_dir)

        # Check that the ONERA dataset exists
        data_dir = f"{data_dir}/{dataset_name}"
        raise_if_no_dir(data_dir)

        # Read the cities to consider
        self.city_names = ["Szada", "Tiszadob"]

        self.city_to_idx = {}
        self.idx_to_city = {}

        # Save paths to all images
        self.img_1_paths = {}
        self.img_2_paths = {}
        self.change_map_paths = {}

        idx = 0

        for city_name in self.city_names:

            # Check that the city directory exists
            city_imgs_dir = f"{data_dir}/{city_name}"
            raise_if_no_dir(city_imgs_dir)

            # There might be several images per city
            sub_city_dir = []
            for _, dirs, _ in os.walk(city_imgs_dir, topdown=False):
                for name in dirs:
                    sub_city_dir.append(name)

            for sub_dir in sub_city_dir:

                sub_city_name = f"{city_name}{sub_dir}"
                sub_imgs_dir = f"{city_imgs_dir}/{sub_dir}"

                if (
                    (self.train and sub_city_name in self.test_names)
                    or (not self.train and city_name not in self.test_names)
                ):

                    continue

                # Check that the images exist
                img_1_filename = f"{sub_imgs_dir}/im1.bmp"
                img_2_filename = f"{sub_imgs_dir}/im2.bmp"

                raise_if_no_file(img_1_filename)
                raise_if_no_file(img_2_filename)

                self.img_1_paths[sub_city_name] = img_1_filename
                self.img_2_paths[sub_city_name] = img_2_filename

                # Check that the label exists
                label_filename = f"{sub_imgs_dir}/gt.bmp"
                raise_if_no_file(label_filename)

                self.change_map_paths[sub_city_name] = label_filename

                # Add city name to list
                self.city_to_idx[sub_city_name] = idx
                self.idx_to_city[idx] = sub_city_name

                idx += 1

        self.n_imgs = len(self.idx_to_city)

    def __len__(self):

        return self.n_imgs

    def __getitem__(self, idx):

        # Match city name
        if isinstance(idx, int):
            city_name = self.idx_to_city[idx]
        else:
            city_name = idx

        # Load the images using PIL
        img_1_pil = Image.open(self.img_1_paths[city_name])
        img_2_pil = Image.open(self.img_2_paths[city_name])

        # Convert the images to numpy in order to concatenate the color channels
        img_1_arr = np.asarray(img_1_pil)
        img_2_arr = np.asarray(img_2_pil)

        # Perform simple equalization to improve visualization
        img_1_scale = simple_equalization_8bit(img_1_arr, percentiles=self.percentiles)
        img_2_scale = simple_equalization_8bit(img_2_arr, percentiles=self.percentiles)

        # Perform histogram matching for correcting colors between images
        img_2_matched = match_histograms(img_2_scale, img_1_scale, channel_axis=-1)

        # Convert back to PIL images for use with PyTorch
        img_1 = Image.fromarray(img_1_scale)
        img_2 = Image.fromarray(img_2_matched)

        # Load the label using PIL
        label = Image.open(self.change_map_paths[city_name])

        # Crop images and label
        img_1_crop, img_2_crop, label_crop = crop_images(
            [img_1, img_2, label], method=self.cropping_method, img_height=self.img_height,
            img_width=self.img_width, left=self.left, top=self.top, right=self.right,
            bottom=self.bottom
        )

        # Apply transforms
        if self.transform is not None:
            img_1_trsf, img_2_trsf, label_trsf = self.transform(
                (img_1_crop, img_2_crop, label_crop)
            )

        # Return sample
        sample = (img_1_trsf, img_2_trsf, label_trsf)

        return sample
