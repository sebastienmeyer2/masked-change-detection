"""Reconstruction for remote sensing data using OmniMAE."""


import argparse


from datasets import OneraReconstructionDataset, SztakiReconstructionDataset
from models import OmniMAEPair, run_cva, run_clustering, load_pretrained_model
from models.omnimae_loss import compute_images_and_loss
from utils.conversion import convert_reconstruction, compute_change_maps
from utils.plotting import plot_reconstruction, plot_change_detection
from utils.misc import set_seed, get_device, str2bool, float_zero_one


def run(
    seed=42,
    data_dir="data",
    dataset_name="OSCD",
    city_name="beirut",
    sub_name="1",
    percentiles=2,
    cropping_method="random",
    left=0.,
    top=0.,
    right=1.,
    bottom=1.,
    video_ordering="1221",
    patch_shape=[2, 16, 16],
    img_height=224,
    img_width=224,
    masking_method="none",
    masking_proportion=0.5,
    window_size=4,
    percentile=95,
    clustering_components=3,
    clustering_threshold=100,
    savefig=True,
    results_dir="results"
):
    """Run an experiment with OmniMAE model and satellite images."""
    # Configuration
    if masking_method == "none":
        masking_proportion = 0

    set_seed(seed=seed)
    device = get_device()

    # Load data
    if dataset_name.lower() in {"onera", "oscd"}:

        dataset = OneraReconstructionDataset(
            data_dir=data_dir, dataset_name="OSCD", img_height=img_height, img_width=img_width,
            percentiles=percentiles, cropping_method=cropping_method, left=left, top=top,
            right=right, bottom=bottom
        )

    elif dataset_name.lower() == "sztaki":

        dataset = SztakiReconstructionDataset(
            data_dir=data_dir, dataset_name="sztaki", percentiles=percentiles,
            cropping_method=cropping_method, img_height=img_height, img_width=img_width, left=left,
            top=top, right=right, bottom=bottom
        )

        city_name = f"{city_name}{sub_name}"

    img_1, img_2, label_trsf = dataset[city_name]

    change_maps = []
    titles = []

    # Compute CVA change map
    change_map_cva = run_cva(
        img_1, img_2, window_size=window_size, percentile=percentile
    )

    change_maps.append(change_map_cva)
    titles.append("cva")

    # Compute clustering change map
    change_maps_clustering = run_clustering(
        img_1, img_2, img_height=img_height, img_width=img_width, components=clustering_components,
        threshold=clustering_threshold
    )

    change_maps.append(change_maps_clustering[0])
    titles.append("clustering")

    # Load pretrained model and do forward pass
    model = load_pretrained_model(device=device)

    omnimae_pair = OmniMAEPair(
        model, video_ordering=video_ordering, patch_shape=patch_shape,
        masking_method=masking_method, masking_proportion=masking_proportion,
        img_height=img_height, img_width=img_width
    )

    img_1 = img_1.to(device)
    img_2 = img_2.to(device)

    output, true_video, mask = omnimae_pair(img_1, img_2)

    # Reconstruct the sequence
    pred_imgs, true_imgs, loss_imgs = compute_images_and_loss(output, true_video, mask)

    # Plots
    results_dir = f"{results_dir}/{dataset_name}/{city_name}"
    filename_prefix = f"{city_name}_{video_ordering}_{masking_method}"
    filename_prefix += f"{masking_proportion}_ws{window_size}_p{percentile}_seed{seed}"

    # Plot reconstruction
    pred_imgs_arr, true_imgs_arr, loss_imgs_bin = convert_reconstruction(
        pred_imgs, true_imgs, loss_imgs, img_height=img_height, img_width=img_width,
        window_size=window_size, percentile=percentile
    )

    plot_reconstruction(
        pred_imgs_arr, true_imgs_arr, loss_imgs_bin, video_ordering=video_ordering,
        savefig=savefig, results_dir=results_dir, filename_prefix=filename_prefix
    )

    # Compute change maps
    change_maps_omnimae = compute_change_maps(
        pred_imgs, true_imgs, loss_imgs, video_ordering=video_ordering, window_size=window_size,
        percentile=percentile
    )

    change_maps.extend(list(change_maps_omnimae))
    titles.extend(["baseline", "reconstruction", "prediction", "both"])

    # Plot change maps
    plot_change_detection(
        change_maps, label_trsf, titles=titles, savefig=savefig, results_dir=results_dir,
        filename_prefix=filename_prefix
    )


if __name__ == "__main__":

    # Command lines
    PARSER_DESC = "Main file to reconstruct satellite images using OmniMAE."
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
        "--data-dir",
        default="data",
        type=str,
        help="""
             Name of the directory where data is stored. Default: "data".
             """
    )

    PARSER.add_argument(
        "--dataset-name",
        default="OSCD",
        type=str,
        choices=["onera", "OSCD", "sztaki"],
        help="""
             Name of the dataset to use. Default: "OSCD".
             """
    )

    PARSER.add_argument(
        "--city-name",
        default="beirut",
        type=str,
        help="""
             Name of the city inside dataset. Default: "beirut".
             """
    )

    PARSER.add_argument(
        "--sub-name",
        default="1",
        type=str,
        help="""
             For SZTAKI dataset, there are several images per city. Default: "1".
             """
    )

    PARSER.add_argument(
        "--percentiles",
        default=2,
        type=float,
        help="""
             Percentiles for saturation in the initial preprocessing of images. Default: 2.
             """
    )

    PARSER.add_argument(
        "--cropping-method",
        default="random",
        type=str,
        help="""
             Cropping method, either "random" for random cropping of 224x224 part of the image or
             "fixed" for a choice using `--left`, `--top`, `--right` and `--bottom`.
             Default: "random".
             """
    )

    PARSER.add_argument(
        "--left",
        default=0.,
        type=float,
        help="""
             Left limit for initial cropping of satellite images. Default: 0.
             """
    )

    PARSER.add_argument(
        "--top",
        default=0.,
        type=float,
        help="""
             Top limit for initial cropping of satellite images. Default: 0.
             """
    )

    PARSER.add_argument(
        "--right",
        default=1.,
        type=float,
        help="""
             Right limit for initial cropping of satellite images. Default: 1.
             """
    )

    PARSER.add_argument(
        "--bottom",
        default=1.,
        type=float,
        help="""
             Bottom limit for initial cropping of satellite images. Default: 1.
             """
    )

    # Model
    PARSER.add_argument(
        "--video-ordering",
        default="1221",
        type=str,
        help="""
             For a given pair of images, where "1" designates the first one and "2" the last one,
             define a sequence of repeated images to be fed to the OmniMAE model as a video.
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

    # Filtering
    PARSER.add_argument(
        "--window-size",
        default=4,
        type=int,
        help="""
             Window size for median filtering of the loss. Use 0 to apply no filtering. Default: 4.
             """
    )

    PARSER.add_argument(
        "--percentile",
        default=95,
        type=float,
        help="""
             Threshold for change point detection on the loss. Default: 95.
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

    # Run experiment
    run(
        seed=ARGS.seed,
        data_dir=ARGS.data_dir,
        dataset_name=ARGS.dataset_name,
        city_name=ARGS.city_name,
        sub_name=ARGS.sub_name,
        percentiles=ARGS.percentiles,
        cropping_method=ARGS.cropping_method,
        left=ARGS.left,
        top=ARGS.top,
        right=ARGS.right,
        bottom=ARGS.bottom,
        video_ordering=ARGS.video_ordering,
        masking_method=ARGS.masking_method,
        masking_proportion=ARGS.masking_proportion,
        window_size=ARGS.window_size,
        percentile=ARGS.percentile,
        savefig=ARGS.savefig,
        results_dir=ARGS.results_dir
    )
