"""Data creation and handling."""


import numpy as np

import torch


RGB_BANDS = [
    "B04",  # RED
    "B03",  # GREEN
    "B02",  # BLUE
]


RGB_NIR_BANDS = [
    "B04",  # RED
    "B03",  # GREEN
    "B02",  # BLUE

    "B08",  # NIR
]


RES20_BANDS = [
    "B04",  # RED
    "B03",  # GREEN
    "B02",  # BLUE

    "B08",  # NIR

    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
]


ALL_BANDS = [
    "B04",  # RED
    "B03",  # GREEN
    "B02",  # BLUE

    "B08",  # NIR

    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",

    "B01",
    "B09",
    "B10",
]


def get_bands(bands_name):

    if bands_name == "rgb":

        bands = RGB_BANDS

    elif bands_name == "nir":

        bands = RGB_NIR_BANDS

    elif bands_name == "res20":

        bands = RES20_BANDS

    elif bands_name == "all":

        bands = ALL_BANDS

    else:

        err_msg = f"Unknown bands name {bands_name}."
        err_msg += """Choose between "rgb", "nir", "res20" and "all"."""
        raise ValueError(err_msg)

    return bands


def apply_fixed_thresholding(
    img,
    threshold=0.1,
    fill_value=1
):
    """Apply a fixed thresholding to loss images."""
    img_thr = img.copy()
    img_thr[img_thr <= threshold] = 0
    img_thr[img_thr >= threshold] = fill_value

    return img_thr


def apply_percentile_thresholding(
    img,
    percentile=95,
    fill_value=1
):
    """Apply a percentile thresholding to loss images."""
    threshold = np.percentile(img, percentile)
    img_pct = apply_fixed_thresholding(img, threshold=threshold, fill_value=fill_value)

    return img_pct


def rescale_zero_one(img):
    """Rescale an image between 0 and 1."""
    if isinstance(img, torch.Tensor):

        img_min = img.min()
        img_max = img.max()

        img = (img - img_min) / (img_max - img_min)

    elif isinstance(img, np.ndarray):

        img_min = np.amin(img)
        img_max = np.amax(img)

        img = (img - img_min) / (img_max - img_min)

    else:

        err_msg = f"Unsupported array type {type(img)} in rescaling."
        raise ValueError(err_msg)

    return img
