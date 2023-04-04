"""Load and prepare OSCD dataset for reconstruction or change detection."""


import os

from math import ceil

import numpy as np
import pandas as pd
from PIL import Image

from scipy.ndimage import zoom

from skimage import io
from skimage.exposure import match_histograms

import torch
from torch.utils.data import Dataset

from tqdm import tqdm


from datasets.transforms import create_reconstruction_transform
from utils.image_processing import simple_equalization_8bit, crop_images
from utils.data import get_bands
from utils.misc import raise_if_no_dir, raise_if_no_file


class OneraReconstructionDataset(Dataset):
    """Dataset used for reconstruction with OmniMAE."""

    def __init__(
        self, data_dir="data", dataset_name="OSCD", imgs_1_dir="imgs_1_rect",
        imgs_2_dir="imgs_2_rect", img_1_base="", img_2_base="", bands_name="rgb",
        transform=None, percentiles=2, cropping_method="random", img_height=224, img_width=224,
        left=0, top=0, right=1, bottom=1
    ):

        # Parameters
        self.bands = get_bands(bands_name)

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
        filename = f"{data_dir}/all.txt"
        raise_if_no_file(filename)

        self.city_names = pd.read_csv(filename).columns

        self.city_to_idx = {}
        self.idx_to_city = {}

        for idx, city_name in enumerate(self.city_names):

            self.city_to_idx[city_name] = idx
            self.idx_to_city[idx] = city_name

        # Save paths to all images
        self.img_1_paths = {band: {} for band in self.bands}
        self.img_2_paths = {band: {} for band in self.bands}
        self.change_map_paths = {}

        for city_name in self.city_names:

            # Check that the city directory exists
            city_imgs_dir = f"{data_dir}/{city_name}"
            raise_if_no_dir(city_imgs_dir)

            # Check that the images directories exist
            city_imgs_1_dir = f"{city_imgs_dir}/{imgs_1_dir}"
            city_imgs_2_dir = f"{city_imgs_dir}/{imgs_2_dir}"

            # Check that the images exist
            for band in self.bands:

                img_1_band = f"{city_imgs_1_dir}/{img_1_base}{band}.tif"
                img_2_band = f"{city_imgs_2_dir}/{img_2_base}{band}.tif"

                raise_if_no_file(img_1_band)
                raise_if_no_file(img_2_band)

                self.img_1_paths[band][city_name] = img_1_band
                self.img_2_paths[band][city_name] = img_2_band

            # Check that the city directory exsits
            city_labels_dir = f"{data_dir}/{city_name}"
            raise_if_no_dir(city_labels_dir)

            # Check that the label exists
            label_filename = f"{city_labels_dir}/cm/{city_name}-cm.tif"
            raise_if_no_file(label_filename)

            self.change_map_paths[city_name] = label_filename

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
        img_1_pil = [Image.open(self.img_1_paths[band][city_name]) for band in self.bands]
        img_2_pil = [Image.open(self.img_2_paths[band][city_name]) for band in self.bands]

        # Convert the images to numpy in order to concatenate the color channels
        img_1_arr = [np.asarray(img) for img in img_1_pil]
        img_2_arr = [np.asarray(img) for img in img_2_pil]
        img_1_cat = np.array(img_1_arr).T
        img_2_cat = np.array(img_2_arr).T

        # Perform simple equalization to improve visualization
        img_1_scale = simple_equalization_8bit(img_1_cat, percentiles=self.percentiles)
        img_2_scale = simple_equalization_8bit(img_2_cat, percentiles=self.percentiles)

        # Perform histogram matching for correcting colors between images
        img_2_matched = match_histograms(img_2_scale, img_1_scale, channel_axis=-1)

        # Convert back to PIL images
        img_1 = Image.fromarray(img_1_scale)
        img_2 = Image.fromarray(img_2_matched)

        # Load the label using PIL (attention: cm is the transpose of img)
        label = Image.open(self.change_map_paths[city_name]).transpose(Image.TRANSPOSE)

        # Map labels from {1, 2} to {0, 255}
        label = 255 * np.asarray(label) - 255
        label = Image.fromarray(label)

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


class OneraChangeDetectionDataset(Dataset):
    """Dataset used for change detection with OSCD data."""

    def __init__(
        self, data_dir="data", dataset_name="OSCD", imgs_1_dir="imgs_1", imgs_2_dir="imgs_2",
        train=True, bands_name="rgb", transform=None, patch_size=224, normalize_imgs=True,
        fp_modifier=10
    ):

        # Parameters
        self.train = train

        self.bands = get_bands(bands_name)

        self.transform = transform

        self.patch_size = patch_size  # compatibility
        self.stride = int(self.patch_size / 2) - 1  # compatibility

        # Check that the data directory exists
        raise_if_no_dir(data_dir)

        # Check that the ONERA dataset exists
        data_dir = f"{data_dir}/{dataset_name}"
        raise_if_no_dir(data_dir)

        # Read the cities to consider
        if train:
            filename = f"{data_dir}/train.txt"
        else:
            filename = f"{data_dir}/test.txt"
        raise_if_no_file(filename)

        self.city_names = pd.read_csv(filename).columns

        self.city_to_idx = {}
        self.idx_to_city = {}

        for idx, city_name in enumerate(self.city_names):

            self.city_to_idx[city_name] = idx
            self.idx_to_city[idx] = city_name

        # Save paths to all images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}

        n_pix = 0
        true_pix = 0

        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []

        for city_name in tqdm(self.city_names, desc="Read OSCD city data"):

            # Check that the city directory exists
            city_imgs_dir = f"{data_dir}/{city_name}"
            raise_if_no_dir(city_imgs_dir)

            # Check that the images directories exist
            city_imgs_1_dir = f"{city_imgs_dir}/{imgs_1_dir}"
            city_imgs_2_dir = f"{city_imgs_dir}/{imgs_2_dir}"
            raise_if_no_dir(city_imgs_1_dir)
            raise_if_no_dir(city_imgs_2_dir)

            # Check that the images exist
            header_1 = os.listdir(city_imgs_1_dir)[0][:-7]
            header_2 = os.listdir(city_imgs_2_dir)[0][:-7]

            img_1_rgb = [f"{city_imgs_1_dir}/{header_1}{band}.tif" for band in self.bands]
            img_2_rgb = [f"{city_imgs_2_dir}/{header_2}{band}.tif" for band in self.bands]
            for filename_1, filename_2 in zip(img_1_rgb, img_2_rgb):
                raise_if_no_file(filename_1)
                raise_if_no_file(filename_2)

            # Load the images
            img_1_io = [io.imread(img) for img in img_1_rgb]
            img_2_io = [io.imread(img) for img in img_2_rgb]

            # Find shape to respect for first images
            img_1_r = img_1_io[0]
            sh = img_1_r.shape

            for i in range(len(self.bands)):

                img_1_band = img_1_io[i]

                sh_band = img_1_band.shape

                if sh_band != sh:

                    # Zoom image
                    img_1_band = zoom(img_1_band, 2)

                    # Crop if necessary
                    img_1_band = img_1_band[:sh[0], :sh[1]]

                    sh_band = img_1_band.shape

                    # Pad if necessary
                    p0 = max(0, sh[0] - sh_band[0])
                    p1 = max(0, sh[1] - sh_band[1])

                    img_1_band = np.pad(img_1_band, ((0, p0), (0, p1)), "edge")

                img_1_io[i] = img_1_band.copy()

            # Find shape to respect for last images
            img_2_r = img_2_io[0]
            sh = img_2_r.shape

            for i in range(len(self.bands)):

                img_2_band = img_2_io[i]

                sh_band = img_2_band.shape

                if sh_band != sh:

                    # Zoom image
                    img_2_band = zoom(img_2_band, 2)

                    # Crop if necessary
                    img_2_band = img_2_band[:sh[0], :sh[1]]

                    sh_band = img_2_band.shape

                    # Pad if necessary
                    p0 = max(0, sh[0] - sh_band[0])
                    p1 = max(0, sh[1] - sh_band[1])

                    img_2_band = np.pad(img_2_band, ((0, p0), (0, p1)), "edge")

                img_2_io[i] = img_2_band.copy()

            # Convert the images to numpy in order to concatenate the color channels
            img_1_cat = np.stack(img_1_io, axis=2).astype("float")
            img_2_cat = np.stack(img_2_io, axis=2).astype("float")

            if normalize_imgs:
                img_1_cat = (img_1_cat - img_1_cat.mean()) / img_1_cat.std()
                img_2_cat = (img_2_cat - img_2_cat.mean()) / img_2_cat.std()

            # Crop if necessary
            sh1 = img_1_cat.shape
            sh2 = img_2_cat.shape
            img_2_cat = np.pad(
                img_2_cat, ((0, sh1[0] - sh2[0]), (0, sh1[1] - sh2[1]), (0, 0)), "edge"
            )

            img_1 = img_1_cat.transpose((2, 0, 1))
            img_2 = img_2_cat.transpose((2, 0, 1))

            # Save the images
            self.imgs_1[city_name] = torch.from_numpy(img_1)
            self.imgs_2[city_name] = torch.from_numpy(img_2)

            city_labels_dir = f"{data_dir}/{city_name}"

            # Check that the label exists
            label_filename = f"{city_labels_dir}/cm/cm.png"
            raise_if_no_file(label_filename)

            # Load the label using PIL (attention: cm is the transpose of img)
            label = io.imread(label_filename, as_gray=True) != 0

            # Save the change map
            self.change_maps[city_name] = label

            # Build patches
            sh = label.shape
            n_pix += np.prod(sh)
            true_pix += label.sum()

            # Calculate the number of patches
            sh = self.imgs_1[city_name].shape
            n1 = ceil((sh[1] - self.patch_size + 1) / self.stride)
            n2 = ceil((sh[2] - self.patch_size + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[city_name] = n_patches_i
            self.n_patches += n_patches_i

            # Generate path coordinates
            for i in range(n1):
                for j in range(n2):

                    # Coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (
                        city_name,
                        [
                            self.stride*i, self.stride*i + self.patch_size, self.stride*j,
                            self.stride*j + self.patch_size
                        ],
                        [self.stride*(i + 1), self.stride*(j + 1)])
                    self.patch_coords.append(current_patch_coords)

        print(self.n_patches, "patches", self.n_patches_per_image)

        # Loss weights
        self.weights = [fp_modifier * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_nb_channels(self):

        nb_channels = len(self.bands)

        return nb_channels

    def get_city_data(self, city_name):

        sample = self.imgs_1[city_name], self.imgs_2[city_name], self.change_maps[city_name]

        return sample

    def __len__(self):

        return self.n_patches

    def __getitem__(self, idx):

        current_patch_coords = self.patch_coords[idx]

        city_name = current_patch_coords[0]
        limits = current_patch_coords[1]

        img_1 = self.imgs_1[city_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        img_2 = self.imgs_2[city_name][:, limits[0]:limits[1], limits[2]:limits[3]]

        label = self.change_maps[city_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1*np.array(label)).float()

        sample = (img_1, img_2, label)

        if self.transform:
            sample = self.transform(sample)

        return sample
