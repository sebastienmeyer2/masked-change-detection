"""Convert pair of images to a fake video for OmniMAE."""


import torch

import einops


def create_video(
    img_1,
    img_2,
    video_ordering="1221"
):
    """Create a video from two images to be repeated."""
    img_1_d = einops.repeat(img_1, "b c h w -> b c t h w", t=1)
    img_2_d = einops.repeat(img_2, "b c h w -> b c t h w", t=1)

    img_list = [img_1_d, img_2_d]

    video_ordering_list = [int(img_nb) - 1 for img_nb in list(video_ordering)]
    video = torch.cat(tuple(img_list[img_idx] for img_idx in video_ordering_list), dim=2)

    # true_video_ordering_list = [
    #     video_ordering_list[2*int(np.ceil(i//2))+1] for i in range(len(video_ordering))
    # ]
    true_video_ordering_list = video_ordering_list
    true_video = torch.cat(tuple(img_list[img_idx] for img_idx in true_video_ordering_list), dim=2)

    return video, true_video


def create_mask(
    batch_size=1,
    video_ordering="1221",
    patch_shape=[2, 16, 16],
    img_height=224,
    img_width=224,
    masking_method="none",
    masking_proportion=0.5
):
    """Create mask."""
    video_length = len(video_ordering)

    nb_mask_time = video_length // patch_shape[0]
    nb_mask_height = img_height // patch_shape[2]
    nb_mask_width = img_width // patch_shape[1]

    if masking_method == "random":

        # Freeze the same amount of patches in batch (mandatory for training to work)
        total_shape = nb_mask_time * nb_mask_width * nb_mask_height
        raw = torch.zeros((total_shape,), dtype=torch.bool)
        raw[int(masking_proportion * total_shape):] = True

        # Assign a random permutation of patches for each element in batch
        mask = torch.empty(
            batch_size, nb_mask_time, nb_mask_width, nb_mask_height, dtype=torch.bool
        )

        for i in range(batch_size):

            ridx = torch.randperm(total_shape)
            mask[i] = torch.reshape(raw[ridx], (nb_mask_time, nb_mask_width, nb_mask_height))

    elif masking_method == "complementary":

        mask = torch.empty(
            batch_size, nb_mask_time // 2, nb_mask_width, nb_mask_height, dtype=torch.bool
        ).bernoulli_(masking_proportion)
        mask = torch.cat((mask, ~mask), dim=1)  # always the same amount of patches

    elif masking_method == "none":

        mask = torch.zeros(
            batch_size, nb_mask_time, nb_mask_width, nb_mask_height, dtype=torch.bool
        )

    else:

        err_msg = f"Unknown masking method {masking_method}."
        err_msg += """ Choose from "random", "complementary" and "none"."""
        raise ValueError(err_msg)

    return mask
