"""Convert output from OmniMAE for visualization."""


import numpy as np


from utils.image_processing import permute_average, permute_clip, median_filter
from utils.data import apply_percentile_thresholding, rescale_zero_one


def convert_reconstruction(
    pred_imgs,
    true_imgs,
    loss_imgs,
    img_height=224,
    img_width=224,
    window_size=4,
    percentile=95
):
    """Convert images from (C, H, W) to (H, W, C) and range 0-255 and apply median filtering."""
    sh = pred_imgs.shape
    t = sh[2]
    c = sh[1]

    pred_imgs_arr = np.zeros((t, img_height, img_width, c), dtype=np.uint8)
    true_imgs_arr = np.zeros((t, img_height, img_width, c), dtype=np.uint8)
    loss_imgs_bin = np.zeros((t, img_height, img_width))

    for i in range(t):

        # Reconstruct the images
        pred_imgs_arr[i] = permute_clip(pred_imgs[0, :, i].detach().cpu().numpy()).copy()
        true_imgs_arr[i] = permute_clip(true_imgs[0, :, i].detach().cpu().numpy()).copy()

        # Average the loss over channels
        loss_img_mean = permute_average(loss_imgs[0, :, i].detach().cpu()).numpy()

        # Median filtering of the loss
        loss_img_flt = median_filter(loss_img_mean, window_size=window_size)

        # Convert loss to binary output
        loss_img_bin = apply_percentile_thresholding(loss_img_flt, percentile=percentile)

        loss_imgs_bin[i] = loss_img_bin.copy()

    return pred_imgs_arr, true_imgs_arr, loss_imgs_bin


def compute_change_maps(
    pred_imgs,
    true_imgs,
    loss,
    video_ordering="1221",
    window_size=4,
    percentile=95
):
    """Compute change maps."""
    # Compute the baseline loss
    first_img = 0
    second_img = 0
    while int(video_ordering[first_img]) == int(video_ordering[second_img]):
        second_img += 1
    base_loss_mean = permute_average(
        (true_imgs[0, :, second_img] - true_imgs[0, :, first_img]).pow(2).detach().cpu().numpy()
    )
    base_loss_mean = rescale_zero_one(base_loss_mean)

    # Compute the reconstruction loss
    img_loss_mean = loss[0].sum(dim=0).sum(dim=0).detach().cpu().numpy()
    img_loss_mean = rescale_zero_one(img_loss_mean)

    # Compute the prediction loss
    t = len(video_ordering)
    i0 = 0
    i1 = 1
    pred_loss_mean = permute_average(
        (pred_imgs[0, :, i0] - pred_imgs[0, :, i1]).pow(2).detach().cpu().numpy()
    )
    while i1 + 1 < t:
        i0 += 1
        i1 += 1
        pred_loss_mean += permute_average(
            (pred_imgs[0, :, i0] - pred_imgs[0, :, i1]).pow(2).detach().cpu().numpy()
        )
    pred_loss_mean = rescale_zero_one(pred_loss_mean)

    # Compute the combination of reconstruction and prediction loss
    both_loss_mean = img_loss_mean + pred_loss_mean
    both_loss_mean = rescale_zero_one(both_loss_mean)

    # Median filtering
    base_loss_flt = median_filter(base_loss_mean, window_size=window_size)
    img_loss_flt = median_filter(img_loss_mean, window_size=window_size)
    pred_loss_flt = median_filter(pred_loss_mean, window_size=window_size)
    both_loss_flt = median_filter(both_loss_mean, window_size=window_size)

    # Binary conversion
    base_loss_bin = apply_percentile_thresholding(base_loss_flt, percentile=percentile)
    img_loss_bin = apply_percentile_thresholding(img_loss_flt, percentile=percentile)
    pred_loss_bin = apply_percentile_thresholding(pred_loss_flt, percentile=percentile)
    both_loss_bin = apply_percentile_thresholding(both_loss_flt, percentile=percentile)

    return base_loss_bin, img_loss_bin, pred_loss_bin, both_loss_bin
