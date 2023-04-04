"""Convolutional and Transformer models for change detection on satellite images."""


import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


from datasets import OneraChangeDetectionDataset
from datasets.transforms import RandomFlipPair, RandomRotPair
from models import (
    OmniMAEPair, FresUNet, OmniMAECNN, OmniMAEFresUNet, load_pretrained_model, train
)
from utils.misc import (
    set_seed, get_device, str2bool, float_zero_one, create_if_no_dir, raise_if_no_dir
)
from utils.plotting import evaluate, plot_model_predictions


def run(
    seed=42,
    bands_name="rgb",
    patch_size=224,
    normalize_imgs=True,
    fp_modifier=10,
    batch_size=12,
    model_name="fresunet",
    patch_shape=[2, 16, 16],
    video_ordering="1221",
    masking_method="none",
    masking_proportion=0.5,
    finetune=True,
    checkpoints_dir="checkpoints",
    load_checkpoint=False,
    save_checkpoint=True,
    n_epochs=50,
    percentiles=0.5,
    clustering_components=3,
    clustering_threshold=100,
    window_size=4,
    percentile=95,
    savefig=True,
    results_dir="results"
):
    """Train a model and compute change maps."""
    # Configuration
    img_height = patch_size
    img_width = patch_size

    if masking_method == "none":
        masking_proportion = 0

    set_seed(seed)
    device = get_device()

    # Prepare datasets
    data_transform = Compose(
        [
            RandomFlipPair(proba_flip=0.5),
            RandomRotPair(do_rotate=True),
        ]
    )

    train_dataset = OneraChangeDetectionDataset(
        train=True, transform=data_transform, bands_name=bands_name, patch_size=patch_size,
        normalize_imgs=normalize_imgs, fp_modifier=fp_modifier
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = OneraChangeDetectionDataset(
        train=False, transform=data_transform, bands_name=bands_name, patch_size=patch_size,
        normalize_imgs=normalize_imgs, fp_modifier=fp_modifier
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    nb_channels = train_dataset.get_nb_channels()

    # Prepare the network
    if model_name == "fresunet":

        model = FresUNet(2*nb_channels, 2)

    elif model_name == "omnimae":

        omnimae_base = load_pretrained_model(device=device)

        model = OmniMAEPair(
            omnimae_base, video_ordering=video_ordering, patch_shape=patch_shape,
            nb_channels=nb_channels, masking_method=masking_method,
            masking_proportion=masking_proportion, img_height=img_height, img_width=img_width
        )

    elif model_name == "omnimaefresunet":

        omnimae_base = load_pretrained_model(device=device)

        omnimae_pair = OmniMAEPair(
            omnimae_base, video_ordering=video_ordering, patch_shape=patch_shape,
            nb_channels=nb_channels, masking_method=masking_method,
            masking_proportion=masking_proportion, img_height=img_height, img_width=img_width
        )

        if finetune:

            omnimae_pair.finetune_first_last_layers()

        model = OmniMAEFresUNet(
            2*nb_channels, 2, nb_channels=nb_channels, patch_shape=patch_shape,
            img_height=img_height, img_width=img_width, omnimae_pair=omnimae_pair
        )

    elif model_name == "omnimaecnn":

        omnimae_base = load_pretrained_model(device=device)

        omnimae_pair = OmniMAEPair(
            omnimae_base, video_ordering=video_ordering, patch_shape=patch_shape,
            nb_channels=nb_channels, masking_method=masking_method,
            masking_proportion=masking_proportion, img_height=img_height, img_width=img_width
        )

        if finetune:

            omnimae_pair.finetune_first_last_layers()

        model = OmniMAECNN(
            omnimae_pair, patch_shape=patch_shape, nb_channels=nb_channels, img_height=img_height,
            img_width=img_width
        )

    else:

        err_msg = f"Unknown model {model_name}."
        err_msg += """Choose between "fresunet", "omnimae" and "omnimaefresunet"."""
        raise ValueError(err_msg)

    # Load checkpoint
    if load_checkpoint:

        raise_if_no_dir(checkpoints_dir)

        model_filename = f"{checkpoints_dir}/{model_name}_{bands_name}"
        if model_name in {"omnimae", "omnimaefresunet"}:
            model_filename += f"_{video_ordering}"
        model_filename += ".pth.tar"

        print(f"Loading checkpoint {model_filename}")
        model.load_state_dict(torch.load(model_filename))

    # Move model to device
    model.to(device)

    # Start training
    if model_name != "omnimae":

        weight = torch.FloatTensor(train_dataset.weights).cuda()
        criterion = nn.NLLLoss(weight=weight)  # to be used with logsoftmax output

        nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of trainable parameters:", nb_params)

        train(model, train_loader, criterion, device=device, n_epochs=n_epochs)

    # Compute scores
    test_results = evaluate(
        model, model_name, test_loader, device=device, percentiles=percentiles,
        clustering_components=clustering_components, clustering_threshold=clustering_threshold,
        video_ordering=video_ordering, window_size=window_size, percentile=percentile
    )

    print(test_results)

    # Plot model predictions
    if savefig:

        # _ = plot_model_predictions(
        #     model, model_name, train_dataset, device=device, percentiles=percentiles,
        #     clustering_components=clustering_components,
        #     clustering_threshold=clustering_threshold, img_height=img_height,
        #     img_width=img_width, bands_name=bands_name, video_ordering=video_ordering,
        #     masking_method=masking_method, masking_proportion=masking_proportion,
        #     window_size=window_size, percentile=percentile, results_dir=results_dir
        # )
        _ = plot_model_predictions(
            model, model_name, test_dataset, device=device, percentiles=percentiles,
            clustering_components=clustering_components, clustering_threshold=clustering_threshold,
            img_height=img_height, img_width=img_width, bands_name=bands_name,
            video_ordering=video_ordering, masking_method=masking_method,
            masking_proportion=masking_proportion, window_size=window_size, percentile=percentile,
            results_dir=results_dir
        )

    # Save checkpoints
    if save_checkpoint:

        create_if_no_dir(checkpoints_dir)

        model_filename = f"{checkpoints_dir}/{model_name}_{bands_name}"
        if model_name in {"omnimae", "omnimaefresunet"}:
            model_filename += f"_{video_ordering}"
        model_filename += ".pth.tar"

        torch.save(model.state_dict(), model_filename)

        print(f"Saving checkpoint {model_filename}")
        model.load_state_dict(torch.load(model_filename))


if __name__ == "__main__":

    # Command lines
    PARSER_DESC = "Main file to perform change detection of satellite images using OmniMAE."
    PARSER = argparse.ArgumentParser(description=PARSER_DESC)

    # Configuration
    PARSER.add_argument(
        "--seed",
        default=42,
        type=int,
        help="""
             Seed to use everywhere for reproducibility. Default: 42.
             """
    )

    # Data
    PARSER.add_argument(
        "--bands-name",
        default="rgb",
        type=str,
        help="""
             Name of the bands to use in the images. Choose from "rgb", "nir", "res20" and "all".
             Default: "rgb".
             """
    )

    PARSER.add_argument(
        "--patch-size",
        default=224,
        type=int,
        help="""
             Size of the cropped images for training. Models based on OmniMAE only support 224.
             Default: 224.
             """
    )

    PARSER.add_argument(
        "--normalize",
        default="True",
        type=str2bool,
        help="""
             If "True", normalize input images. Default: "True".
             """
    )

    PARSER.add_argument(
        "--fp-modifier",
        default=10,
        type=int,
        help="""
             Should be used with caution. Default: 10.
             """
    )

    # Model
    PARSER.add_argument(
        "--model-name",
        default="fresunet",
        type=str,
        choices=["fresunet", "omnimae", "omnimaecnn", "omnimaefresunet"],
        help="""
             Name of the model to use for change detection. Choose from "fresunet", "omnimae",
             "omnimaecnn" and "omnimaefresunet". Default "fresunet".
             """
    )

    PARSER.add_argument(
        "--video-ordering",
        default="1221",
        type=str,
        help="""
             For a given pair of images, where "1" designates the first one and "2" the last one,
             define a sequence of repeated images to be fed to the OmniMAE model as a video.
             only used for OmniMAE models.
             Default: "1221".
             """
    )

    PARSER.add_argument(
        "--masking-method",
        default="none",
        type=str,
        help="""
             Masking method, either "random", "complementary" or "none" to mask out the whole
             images. When method is "random" or "complementary", set the proportion of masked
             patches (for "complementary", in the first image) using the `--masking-proportion`
             argument. Default: "none".
             """
    )

    PARSER.add_argument(
        "--masking-proportion",
        default=0.5,
        type=float_zero_one,
        help="""
             When `--masking-method` is set to "random" or "complementary", set the proportion of
             masked patches, i.e. the proportion of patches which are not fed to the model.
             Default: 0.5.
             """
    )

    PARSER.add_argument(
        "--finetune",
        default="True",
        type=str2bool,
        help="""
             If "True", finetune first and last layers of OmniMAE. Default: "True".
             """
    )

    PARSER.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        type=str,
        help="""
             Name of the directory where checkpoints are loaded and / or saved.
             Default: "checkpoints".
             """
    )

    PARSER.add_argument(
        "--load-checkpoint",
        default="False",
        type=str2bool,
        help="""
             If "True", load checkpoint from disk. Default: "False".
             """
    )

    PARSER.add_argument(
        "--save-checkpoint",
        default="True",
        type=str2bool,
        help="""
             If "True", save checkpoint to disk. Default: "True".
             """
    )

    # Training
    PARSER.add_argument(
        "--batch-size",
        default=12,
        type=int,
        help="""
             Batch size for training the model. Default: 12.
             """
    )

    PARSER.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="""
             Number of training epochs. Default: 50.
             """
    )

    # Save
    PARSER.add_argument(
        "--savefig",
        default="True",
        type=str2bool,
        help="""
             If "True", figures will be saved on disk. Default: "True".
             """
    )

    PARSER.add_argument(
        "--results-dir",
        default="results",
        type=str,
        help="""
             Name of the directory where figures are stored. Default: "results".
             """
    )

    # End of command lines
    ARGS = PARSER.parse_args()

    # Run training
    run(
        seed=ARGS.seed,
        bands_name=ARGS.bands_name,
        patch_size=ARGS.patch_size,
        normalize_imgs=ARGS.normalize,
        fp_modifier=ARGS.fp_modifier,
        batch_size=ARGS.batch_size,
        model_name=ARGS.model_name,
        video_ordering=ARGS.video_ordering,
        masking_method=ARGS.masking_method,
        masking_proportion=ARGS.masking_proportion,
        finetune=ARGS.finetune,
        checkpoints_dir=ARGS.checkpoints_dir,
        load_checkpoint=ARGS.load_checkpoint,
        save_checkpoint=ARGS.save_checkpoint,
        n_epochs=ARGS.epochs,
        savefig=ARGS.savefig,
        results_dir=ARGS.results_dir
    )
