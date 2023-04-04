"""Change Vector Analysis."""


import numpy as np

import torch


from utils.data import apply_percentile_thresholding, rescale_zero_one
from utils.image_processing import median_filter


def run_cva(img_1, img_2, window_size=4, percentile=95):

    # Convert to arrays if needed
    if isinstance(img_1, torch.Tensor):

        img_1 = img_1.squeeze().detach().cpu().numpy()
        img_2 = img_2.squeeze().detach().cpu().numpy()

    # Transpose if needed
    if img_1.shape[0] <= 13:  # channels first

        img_1 = img_1.transpose((1, 2, 0))
        img_2 = img_2.transpose((1, 2, 0))

    # Normalize the features (separate for both images)
    img_1_mean = np.mean(img_1, axis=(0, 1))
    img_1_std = np.std(img_1, axis=(0, 1))
    img_1_scale = (img_1 - img_1_mean) / img_1_std

    img_2_mean = np.mean(img_2, axis=(0, 1))
    img_2_std = np.std(img_2, axis=(0, 1))
    img_2_scale = (img_2 - img_2_mean) / img_2_std

    # Compute distance between the two normalized vectors
    img_diff = np.abs(img_1_scale - img_2_scale)

    change_map = np.linalg.norm(img_diff, axis=2)

    # Rescale change map
    change_map_scale = rescale_zero_one(change_map)

    # Median filtering
    change_map_flt = median_filter(change_map_scale, window_size=window_size)

    # Binary conversion
    change_map_bin = apply_percentile_thresholding(change_map_flt, percentile=percentile)

    return change_map_bin
