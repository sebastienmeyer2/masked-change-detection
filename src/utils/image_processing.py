"""Utilitary functions to process images, either arrays or tensors."""


import numpy as np
from PIL import Image

import torch


def simple_equalization_8bit(img, percentiles=2):
    """Simple 8-bit requantization by linear stretching.

    Args:
        im (np.array): image to requantize
        percentiles (float): percentage of the darkest and brightest pixels to saturate

    Returns:
        img (np.array): numpy array with the quantized uint8 image
    """
    # Saturation
    min_value, max_value = np.percentile(img[np.isfinite(img)], (percentiles, 100 - percentiles))
    img = np.clip(img, min_value, max_value)

    # Scale image and convert to integers
    img = (img - min_value) / (max_value - min_value) * 255
    img = img.astype(np.uint8)

    return img


def crop_images(
    img_list,
    method="random",
    img_height=224,
    img_width=224,
    left=0.,
    top=0.,
    right=1.,
    bottom=1.
):
    """Crop PIL images using proportions of height and width.

    Recall that for PIL images, (0, 0) is at the upper left of the image.
    """
    img_1 = img_list[0]

    if isinstance(img_1, Image.Image):
        width, height = img_1.size
    elif isinstance(img_1, torch.Tensor):
        width, height = img_1.shape[1:]
    else:
        width, height = img_1.shape[:-1]

    if method == "random":

        left = np.random.randint(0, width-img_width, size=1)[0]
        top = np.random.randint(0, height-img_height, size=1)[0]
        right = left + img_width
        bottom = top + img_height

    else:

        assert 1. >= left >= 0., "Choose left boundary as a float between 0 and 1."
        assert 1. >= top >= 0., "Choose top boundary as a float between 0 and 1."
        assert 1. >= right >= 0., "Choose right boundary as a float between 0 and 1."
        assert 1. >= bottom >= 0., "Choose bottom boundary as a float between 0 and 1."

        assert left < right, "Choose left boundary smaller than right boundary."
        assert top < bottom, "Choose top boundary smaller than bottom boundary."

        left = int(np.round(left * width))
        top = int(np.round(top * height))
        right = int(np.round(right * width))
        bottom = int(np.round(bottom * height))

    if isinstance(img_1, Image.Image):
        img_cropped = [img.crop((left, top, right, bottom)) for img in img_list]
    elif isinstance(img_1, torch.Tensor):
        img_cropped = [
            img[:, left:right, :][:, :, top:bottom]
            if len(img.shape) == 3 else img[left:right, :][:, top:bottom]
            for img in img_list
        ]
    else:
        img_cropped = [
            img[left:right, :, :][:, top:bottom, :]
            if len(img.shape) == 3 else img[left:right, :][:, top:bottom]
            for img in img_list
        ]

    return img_cropped


def permute_clip(img):
    """Permute channel dimension and clip values in 0-255 range."""
    if isinstance(img, torch.Tensor):

        img = torch.einsum("chw->hwc", img)
        img = torch.clip(img * 255, 0, 255).int()

    elif isinstance(img, np.ndarray):

        img = img.transpose(1, 2, 0)
        img = np.clip(255. * img, 0, 255).astype(np.uint8)

    else:

        err_msg = f"Unsupported array type {type(img)} in permutation and clipping."
        raise ValueError(err_msg)

    return img


def permute_average(img):
    """Permute channel dimension and average values over channels."""
    if isinstance(img, torch.Tensor):

        img = torch.einsum("chw->hwc", img)
        img = torch.mean(img, dim=-1)

    elif isinstance(img, np.ndarray):

        img = img.transpose(1, 2, 0)
        img = np.mean(img, axis=-1)

    else:

        err_msg = f"Unsupported array type {type(img)} in permutation and averaging."
        raise ValueError(err_msg)

    return img


def median_filter(img, window_size=0):
    """Median filtering of a given image."""
    n = window_size  # shorter
    img_ndim = img.ndim

    if img_ndim == 3:

        h, w, c = img.shape
        img_type = img.dtype

        filtered_img = np.zeros((h, w, c), dtype=img_type)

        for i in range(h):
            for j in range(w):

                window = img[max(0, i-n):min(i+n+1, h), max(0, j-n):min(j+n+1, w)]

                for k in range(c):
                    filtered_img[i][j][k] = np.median(window[:, :, k])

    elif img_ndim == 2:

        h, w = img.shape
        img_type = img.dtype

        filtered_img = np.zeros((h, w), dtype=img_type)

        for i in range(h):
            for j in range(w):

                window = img[max(0, i-n):min(i+n+1, h), max(0, j-n):min(j+n+1, w)]

                filtered_img[i][j] = np.median(window)

    else:

        err_msg = f"Unsupported input number of dimensions {img_ndim}."
        raise ValueError(err_msg)

    return filtered_img
