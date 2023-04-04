"""Compute the loss, patchify or unpatchify results from OmniMAE."""


import numpy as np

import torch

import einops


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def patchify(imgs, patch_shape=[2, 16, 16], nb_channels=3):
    """Adapted from omnivision.losses.mae_loss.MAELoss.patchify()."""
    assert imgs.shape[-2] == imgs.shape[-1]  # Spatial dimensions match up

    # Add a dummy time dimension to 2D patches for consistency
    # Since it is 1, it will not affect the final number of patches
    if len(patch_shape) == 2:
        patch_shape = [1,] + patch_shape
        imgs = imgs.unsqueeze(-3)

    assert imgs.ndim - 2 == len(patch_shape)  # except batch and channel dims
    for i in range(1, len(patch_shape) + 1):
        assert (
            imgs.shape[-i] % patch_shape[-i] == 0
        ), f"image shape {imgs.shape} & patch shape {patch_shape} mismatch at index {i}"

    p = patch_shape[-3]
    q = patch_shape[-2]
    r = patch_shape[-1]
    t = imgs.shape[-3] // p  # temporality
    h = imgs.shape[-2] // q  # height
    w = imgs.shape[-1] // r  # width
    c = nb_channels

    x = imgs.reshape(shape=(imgs.shape[0], c, t, p, h, q, w, r))
    x = torch.einsum("nctphqwr->nthwpqrc", x)
    patchified_imgs = x.reshape(shape=(imgs.shape[0], t * h * w, p * q * r, c))

    return patchified_imgs


def unpatchify(
    patchified_imgs, patch_shape=[2, 16, 16], nb_channels=3, img_height=224, img_width=224
):
    """Our own function to reverse patchify.

    Adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py#L109.
    """
    p = patch_shape[-3]  # temporality
    q = patch_shape[-2]  # height
    r = patch_shape[-1]  # width
    h = img_height // q
    w = img_width // r
    t = patchified_imgs.shape[1] // (h*w)
    c = nb_channels

    x = patchified_imgs.reshape((patchified_imgs.shape[0], t, h, w, p, q, r, c))
    x = torch.einsum("nthwpqrc->nctphqwr", x)
    x = x.reshape((patchified_imgs.shape[0], c, t, p, h, q, w, r))
    imgs = x.reshape((patchified_imgs.shape[0], c, t * p, h * q, w * r))

    return imgs


def compute_images_and_loss(
    pred_imgs_np, true_imgs_ng, mask, replace_mask=False, patch_shape=[2, 16, 16], nb_channels=3,
    img_height=224, img_width=224, norm_pix_loss=True, norm_pix_per_channel=True,
    trsf_mean=IMAGENET_MEAN, trsf_std=IMAGENET_STD
):
    """Our own function to convert the output of pretrained model to images.

    Adapted from omnivision.losses.mae_loss.MAELoss.compute_mae_loss().

    The suffix "n" means normalized and the suffix "un" means unnormalized.
    The suffix "g" means globally and the suffix "p" means per patch.
    OmniMAE outputs images which are normalized per patch wrt the input images.
    """
    # Reverse the global normalization of the true images
    img_mean = (
        torch.as_tensor(trsf_mean)
        .to(true_imgs_ng.device)
        .reshape([1, -1] + [1] * (true_imgs_ng.ndim - 2))
    )
    img_std = (
        torch.as_tensor(trsf_std)
        .to(true_imgs_ng.device)
        .reshape([1, -1] + [1] * (true_imgs_ng.ndim - 2))
    )
    true_imgs_ung = true_imgs_ng * img_std + img_mean

    # Replicate the true images - deprecated
    img_shape = true_imgs_ung.shape
    if len(img_shape) == 4:  # missing time dimension
        true_imgs_ung = einops.repeat(true_imgs_ung, "b c h w -> b c t h w", t=2)
        true_imgs_ung = true_imgs_ung.to(true_imgs_ng.device)
    elif len(img_shape) == 5 and img_shape[2] == 1:  # single image to replicate
        true_imgs_ung = true_imgs_ung[:, :, 0, :, ...]
        true_imgs_ung = einops.repeat(true_imgs_ung, "b c h w -> b c t h w", t=2)
        true_imgs_ung = true_imgs_ung.to(true_imgs_ng.device)

    # Patchify true images
    target = patchify(true_imgs_ung, patch_shape=patch_shape, nb_channels=nb_channels)

    # Squeeze back channels from linear output
    channels_dim = 1
    nb_channels = true_imgs_ung.shape[channels_dim]

    pred_imgs_np = pred_imgs_np.reshape(
        (*pred_imgs_np.shape[:-1], pred_imgs_np.shape[-1] // nb_channels, nb_channels)
    )

    # Unnormalize predicted patches and compute the loss
    patches_dim = -2

    if norm_pix_loss:

        if not norm_pix_per_channel:

            # Merge the channel with patches and compute mean over all channels of all patches
            # Else, will compute a mean for each channel separately
            target = torch.flatten(target, patches_dim)
            patches_dim = -1

        mean = target.mean(dim=patches_dim, keepdim=True)
        var = target.var(dim=patches_dim, keepdim=True)

        pred_imgs_unp = (var**0.5) * pred_imgs_np + mean

        target_np = (target - mean) / (var + 1.0e-6) ** 0.5

    loss_imgs = (pred_imgs_np - target_np) ** 2

    # Replace non-masked patches to those from original images for better visualization
    if replace_mask:

        mask_flatten = mask.reshape(mask.shape[0], -1)

        pred_imgs_unp[mask_flatten] = torch.clone(target[mask_flatten])
        loss_imgs[mask_flatten] = 0.

    # Unpatchify the predicted images
    pred_imgs_unp = unpatchify(
        pred_imgs_unp, patch_shape=patch_shape, nb_channels=nb_channels, img_height=img_height,
        img_width=img_width
    )
    loss_imgs = unpatchify(
        loss_imgs, patch_shape=patch_shape, nb_channels=nb_channels, img_height=img_height,
        img_width=img_width
    )

    return pred_imgs_unp, true_imgs_ung, loss_imgs
