"""Utilitary functions to plot and save results."""


import os

import warnings

import numpy as np

from skimage.exposure import match_histograms

import torch

from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from models import run_clustering, run_cva
from models.omnimae_loss import compute_images_and_loss
from utils.image_processing import simple_equalization_8bit
from utils.conversion import compute_change_maps


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18
})


MODELS_NAMES_TO_TITLES = {
    "baseline": r"$\mathbf{Baseline \ loss}$",
    "reconstruction": r"$\mathbf{Reconstruction \ loss}$",
    "prediction": r"$\mathbf{Prediction \ loss}$",
    "both": r"$\mathbf{Both \ losses}$",
    "close": r"$\mathbf{Closing}$",
    "open": r"$\mathbf{Opening}$",
    "clustering": r"$\mathbf{Clustering}$",
    "cva": r"$\mathbf{CVA}$",
    "fresunet": r"$\mathbf{FresUNet}$",
    "omnimae": r"$\mathbf{OmniMAE}$",
    "omnimaecnn": r"$\mathbf{OmniMAE + CNN}$",
    "omnimaefresunet": r"\mathbf{OmniMAE + FresUNet}$"
}


def save_or_plot(
    fig,
    savefig=True,
    results_dir="results",
    filename_prefix="",
    filename="out.png"
):
    """Save or plot given figure."""
    if savefig:

        results_dir_split = results_dir.split("/")

        results_dir = ""

        for spl in results_dir_split:

            if len(results_dir) == 0:
                results_dir = spl
            else:
                results_dir = f"{results_dir}/{spl}"

            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

        # Add prefix
        if len(filename_prefix) > 0:
            filename = f"{filename_prefix}_{filename}"

        # Create the results directory where data will be stored
        fig.savefig(f"{results_dir}/{filename}", facecolor="white")

        # Clear figure
        plt.close(fig)


def plot_reconstruction(
    pred_imgs,
    true_imgs,
    loss_imgs_bin,
    video_ordering="1221",
    savefig=True,
    results_dir="results",
    filename_prefix="",
    filename="reconstruction.png"
):
    """Plot a comparison of reconstructed images and their loss."""
    t = len(video_ordering)

    nrows = 3
    ncols = t

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), constrained_layout=True
    )

    for i in range(t):

        current_img = video_ordering[i]

        axs[0, i].imshow(true_imgs[i])
        axs[0, i].set_title(f"True image {current_img}")
        axs[0, i].axis("off")

        axs[1, i].imshow(pred_imgs[i])
        axs[1, i].set_title(f"Reconstructed image {current_img}")
        axs[1, i].axis("off")

        axs[2, i].imshow(loss_imgs_bin[i], cmap="gray")
        axs[2, i].set_title(f"Loss image {current_img}")
        axs[2, i].axis("off")

    save_or_plot(
        fig, savefig=savefig, results_dir=results_dir, filename_prefix=filename_prefix,
        filename=filename
    )


def plot_change_detection(
    change_maps,
    label,
    titles=None,
    savefig=True,
    results_dir="results",
    filename_prefix="",
    filename="change_point_detection.png"
):
    """Plot a comparison of predicted change maps and true labels."""
    if not isinstance(change_maps, list):
        change_maps = [change_maps]

    ncols = len(change_maps)

    fig, axs = plt.subplots(
        ncols=ncols, figsize=(6*ncols, 6), constrained_layout=True, squeeze=False
    )

    for i, change_map in enumerate(change_maps):

        label_flatten = label.flatten()
        change_map_flatten = change_map.flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            f1 = f1_score(label_flatten, change_map_flatten, zero_division=0)
            cg_acc = accuracy_score(
                label_flatten[label_flatten > 0.5], change_map_flatten[label_flatten > 0.5]
            )
            ncg_acc = accuracy_score(
                label_flatten[label_flatten < 0.5], change_map_flatten[label_flatten < 0.5]
                )
            acc = accuracy_score(label_flatten, change_map_flatten)

        if isinstance(label, torch.Tensor):
            label_arr = label.detach().cpu().numpy()
        else:
            label_arr = label.copy()
        label_comp = label_arr.copy()
        label_comp[label_arr > 0.5] = 2  # positives
        label_comp[change_map > label_arr] = 1  # false positives
        label_comp[change_map < label_arr] = 3  # false negatives

        palette = np.array(
            [
                [255, 255, 255],  # white for true negatives
                [255, 0, 0],  # red for false positives
                [0, 255, 0],  # green for true positives
                [0, 0, 255],  # blue for false negatives
            ]
        )

        label_rgb = palette[label_comp]

        axs[0, i].imshow(label_rgb, interpolation="none")

        # Add a legend
        patches = [
            mpatches.Patch(color="white", label="TN"),
            mpatches.Patch(color="blue", label="FN"),
            mpatches.Patch(color="red", label="FP"),
            mpatches.Patch(color="green", label="TP")
        ]
        axs[0, i].legend(handles=patches, loc="upper left", fontsize=22)

        # Remove x and y ticks
        axs[0, i].xaxis.set_tick_params(labelbottom=False)
        axs[0, i].yaxis.set_tick_params(labelleft=False)
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])

        if titles is None:
            title = ""
        else:
            title = f"{MODELS_NAMES_TO_TITLES.get(titles[i], titles[i])}" + "\n"
        title += f"Change acc.: {cg_acc:.2f}\n"
        title += f"No change acc.: {ncg_acc:.2f}\n"
        title += f"Accuracy: {acc:.2f}\n"
        title += f"F1-score: {f1:.2f}"
        axs[0, i].set_title(title, fontsize=22)

    save_or_plot(
        fig, savefig=savefig, results_dir=results_dir, filename_prefix=filename_prefix,
        filename=filename
    )


def plot_summary(
    true_first,
    true_last,
    change_map,
    label,
    title=None,
    savefig=True,
    results_dir="results",
    filename_prefix="",
    filename="summary.png"
):
    """Plot a comparison of predicted change points and true labels."""
    nrows = 1
    ncols = 4

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(3*ncols, 6*nrows), constrained_layout=True
    )

    for j in range(ncols):
        axs[j].axis("off")

    # Plot images
    axs[0].imshow(true_first)
    axs[0].set_title("True first image")

    axs[1].imshow(true_last)
    axs[1].set_title("True last image")

    axs[2].imshow(label, cmap="gray")
    axs[2].set_title("True change map")

    # Plot change map
    label_flatten = label.flatten()
    change_map_flatten = change_map.flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f1 = f1_score(label_flatten, change_map_flatten, zero_division=0)
        cg_acc = accuracy_score(
            label_flatten[label_flatten > 0.5], change_map_flatten[label_flatten > 0.5]
        )
        ncg_acc = accuracy_score(
            label_flatten[label_flatten < 0.5], change_map_flatten[label_flatten < 0.5]
            )
        acc = accuracy_score(label_flatten, change_map_flatten)

    if isinstance(label, torch.Tensor):
        label_arr = label.detach().cpu().numpy()
    else:
        label_arr = label.copy()
    label_comp = label_arr.copy()
    label_comp[label_arr > 0.5] = 2  # positives
    label_comp[change_map > label_arr] = 1  # false positives
    label_comp[change_map < label_arr] = 3  # false negatives

    palette = np.array(
        [
            [255, 255, 255],  # white for true negatives
            [255, 0, 0],  # red for false positives
            [0, 255, 0],  # green for true positives
            [0, 0, 255],  # blue for false negatives
        ]
    )

    label_rgb = palette[label_comp]

    axs[3].imshow(label_rgb, interpolation="none")

    # Add a legend
    patches = [
        mpatches.Patch(color="white", label="TN"),
        mpatches.Patch(color="blue", label="FN"),
        mpatches.Patch(color="red", label="FP"),
        mpatches.Patch(color="green", label="TP")
    ]
    axs[3].legend(handles=patches, loc="upper left")

    # Remove x and y ticks
    axs[3].xaxis.set_tick_params(labelbottom=False)
    axs[3].yaxis.set_tick_params(labelleft=False)
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    axs[3].set_title("Predicted change map")

    if title is None:
        title = ""
    else:
        title = f"{MODELS_NAMES_TO_TITLES.get(title, title)}" + "\n"
    title += f"Change acc.: {cg_acc:.2f}\n"
    title += f"No change acc.: {ncg_acc:.2f}\n"
    title += f"Accuracy: {acc:.2f}\n"
    title += f"F1-score: {f1:.2f}"
    fig.suptitle(title)

    save_or_plot(
        fig, savefig=savefig, results_dir=results_dir, filename_prefix=filename_prefix,
        filename=filename
    )


def plot_model_predictions(
    model,
    model_name,
    dataset,
    device="cuda",
    img_height=224,
    img_width=224,
    percentiles=0.5,
    clustering_components=3,
    clustering_threshold=100,
    bands_name="rgb",
    video_ordering="1221",
    masking_method="none",
    masking_proportion=0.5,
    results_dir="results",
    window_size=4,
    percentile=95
):

    model.eval()

    if model_name == "omnimae":

        models_names = ["cva", "clustering", "baseline", "reconstruction", "prediction", "omnimae"]

    else:

        models_names = ["cva", "clustering", model_name]

    tot_count = {mn: 0 for mn in models_names}

    n = 2  # number of classes
    class_correct = {mn: [0.0 for i in range(n)] for mn in models_names}
    class_total = {mn: [0.0 for i in range(n)] for mn in models_names}
    class_acc = {mn: [0.0 for i in range(n)] for mn in models_names}

    tp = {mn: 0 for mn in models_names}
    tn = {mn: 0 for mn in models_names}
    fp = {mn: 0 for mn in models_names}
    fn = {mn: 0 for mn in models_names}

    pbar = tqdm(dataset.city_names, desc="PLotting change maps")

    for city_name in pbar:

        pbar.set_description(f"Plotting change maps for {city_name}")

        # Retrieve full city images
        img_1_full, img_2_full, label_full = dataset.get_city_data(city_name)

        sh = label_full.shape

        # We will pass over all img_height*img_width images
        for ii in range(sh[0] // img_width):

            for jj in range(sh[1] // img_height):

                xmin = ii * img_width
                xmax = min((ii+1) * img_width, sh[0])
                ymin = jj * img_height
                ymax = min((jj+1) * img_height, sh[1])

                img_1 = img_1_full[:, xmin:xmax, ymin:ymax].float()
                img_2 = img_2_full[:, xmin:xmax, ymin:ymax].float()
                label = torch.from_numpy(1.0*label_full[xmin:xmax, ymin:ymax]).long().numpy()

                # Move to device
                img_1 = img_1.to(device)
                img_2 = img_2.to(device)

                # Predict change map
                res = model(img_1.unsqueeze(0), img_2.unsqueeze(0))

                pred_labels = []

                # Convert images to arrays
                img_1 = img_1.detach().cpu().numpy().transpose((1, 2, 0))
                img_2 = img_2.detach().cpu().numpy().transpose((1, 2, 0))

                img_1 = simple_equalization_8bit(img_1, percentiles=percentiles)
                img_2 = simple_equalization_8bit(img_2, percentiles=percentiles)

                img_2_matched = match_histograms(img_2, img_1, channel_axis=-1)

                # Compute change vector analysis label
                cva_label = run_cva(
                    img_1, img_2_matched, window_size=window_size, percentile=percentile
                )
                pred_labels.append(cva_label)

                # Compute clustering label
                cluster_label, _, _ = run_clustering(
                    img_1, img_2_matched, components=clustering_components,
                    threshold=clustering_threshold
                )
                pred_labels.append(cluster_label)

                # Compute model labels
                if model_name == "omnimae":

                    output, true_video, mask = res

                    pred_imgs, true_imgs, loss_imgs = compute_images_and_loss(
                        output, true_video, mask
                    )

                    model_labels = compute_change_maps(
                        pred_imgs, true_imgs, loss_imgs, video_ordering=video_ordering,
                        window_size=window_size, percentile=percentile
                    )

                    pred_labels.extend(list(model_labels))

                else:

                    _, model_label = torch.max(res.data, 1)
                    pred_labels.append(model_label.squeeze().detach().cpu().numpy())

                # Compute scores
                for i, (mn, pred_label) in enumerate(zip(models_names, pred_labels)):

                    tot_count[mn] += np.prod(label.shape)

                    c = pred_label == label
                    for i in range(c.shape[0]):
                        for j in range(c.shape[1]):
                            pct = int(label[i, j])
                            class_correct[mn][pct] += c[i, j]
                            class_total[mn][pct] += 1

                    pr = pred_label > 0
                    gt = label > 0

                    tp[mn] += np.logical_and(pr, gt).sum()
                    tn[mn] += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                    fp[mn] += np.logical_and(pr, np.logical_not(gt)).sum()
                    fn[mn] += np.logical_and(np.logical_not(pr), gt).sum()

                # Save change map
                city_results_dir = f"{results_dir}/OSCD/{city_name}"
                filename_prefix = f"{city_name}{ii}{jj}_{model_name}_{bands_name}_{video_ordering}"
                filename_prefix += f"_{masking_method}{masking_proportion}"

                plot_change_detection(
                    pred_labels, label, titles=models_names, results_dir=city_results_dir,
                    filename_prefix=filename_prefix
                )

                plot_summary(
                    img_1[:, :, :3], img_2[:, :, :3], pred_labels[-3], label, title=model_name,
                    results_dir=city_results_dir, filename_prefix=filename_prefix
                )

    test_acc = {}

    prec = {}
    rec = {}
    f_score = {}

    prec_nc = {}
    rec_nc = {}
    f_score_nc = {}

    for mn in models_names:

        test_acc[mn] = 100.0 * (tp[mn] + tn[mn]) / tot_count[mn]

        for i in range(n):
            class_acc[mn][i] = 100.0 * class_correct[mn][i] / max(class_total[mn][i], 1e-5)
            class_acc[mn][i] = float(class_acc[mn][i])

        model_tp = tp[mn]
        model_tn = tn[mn]
        model_fp = fp[mn]
        model_fn = fn[mn]

        prec[mn] = model_tp / (model_tp + model_fp)
        rec[mn] = model_tp / (model_tp + model_fn)
        f_score[mn] = 2.0 * (prec[mn] * rec[mn]) / (prec[mn] + rec[mn])

        prec_nc[mn] = model_tn / (model_tn + model_fn)
        rec_nc[mn] = model_tn / (model_tn + model_fp)
        f_score_nc[mn] = 2.0 * (prec_nc[mn] * rec_nc[mn]) / (prec_nc[mn] + rec_nc[mn])

    metrics_results = {
        "accuracy": test_acc,
        "class_accuracy": class_acc,
        "change precision": prec,
        "change recall": rec,
        "change f1-score": f_score,
        "no change precision": prec_nc,
        "no change recall": rec_nc,
        "no change f1-score": f_score_nc
    }

    return metrics_results


def evaluate(
    model,
    model_name,
    test_loader,
    device="cuda",
    percentiles=0.5,
    clustering_components=3,
    clustering_threshold=100,
    video_ordering="1221",
    window_size=4,
    percentile=95
):

    model.eval()

    if model_name == "omnimae":

        models_names = ["baseline", "reconstruction", "prediction", "omnimae", "clustering", "cva"]

    else:

        models_names = [model_name, "clustering", "cva"]

    tot_count = {mn: 0 for mn in models_names}

    n = 2  # number of classes
    class_correct = {mn: [0.0 for i in range(n)] for mn in models_names}
    class_total = {mn: [0.0 for i in range(n)] for mn in models_names}
    class_acc = {mn: [0.0 for i in range(n)] for mn in models_names}

    tp = {mn: 0 for mn in models_names}
    tn = {mn: 0 for mn in models_names}
    fp = {mn: 0 for mn in models_names}
    fn = {mn: 0 for mn in models_names}

    pbar = tqdm(test_loader, desc="Computing scores")

    for batch in pbar:

        # Read data
        img_1, img_2, label = batch

        # Move to device
        img_1 = img_1.float().to(device)
        img_2 = img_2.float().to(device)
        label = label.squeeze().long().numpy()

        # Predict change map
        res = model(img_1, img_2)

        if model_name == "omnimae":

            output, true_video, mask = res

            pred_imgs, true_imgs, loss_imgs = compute_images_and_loss(
                output, true_video, mask
            )

            pred_labels = list(compute_change_maps(
                pred_imgs, true_imgs, loss_imgs, video_ordering=video_ordering,
                window_size=window_size, percentile=percentile
            ))

        else:

            _, res = torch.max(res.data, 1)
            pred_labels = [res.squeeze().detach().cpu().numpy()]

        # Convert images to arrays
        img_1 = img_1.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
        img_2 = img_2.squeeze().detach().cpu().numpy().transpose((1, 2, 0))

        img_1 = simple_equalization_8bit(img_1, percentiles=percentiles)
        img_2 = simple_equalization_8bit(img_2, percentiles=percentiles)

        img_2_matched = match_histograms(img_2, img_1, channel_axis=-1)

        # Compute clustering label
        cluster_label, _, _ = run_clustering(
            img_1, img_2_matched, components=clustering_components,
            threshold=clustering_threshold
        )
        pred_labels.append(cluster_label)

        # Compute change vector analysis label
        cva_label = run_cva(img_1, img_2_matched, window_size=window_size, percentile=percentile)
        pred_labels.append(cva_label)

        # Compute scores
        for i, (mn, pred_label) in enumerate(zip(models_names, pred_labels)):

            tot_count[mn] += np.prod(label.shape)

            c = pred_label == label
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    pct = int(label[i, j])
                    class_correct[mn][pct] += c[i, j]
                    class_total[mn][pct] += 1

            pr = pred_label > 0
            gt = label > 0

            tp[mn] += np.logical_and(pr, gt).sum()
            tn[mn] += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
            fp[mn] += np.logical_and(pr, np.logical_not(gt)).sum()
            fn[mn] += np.logical_and(np.logical_not(pr), gt).sum()

    test_acc = {}

    prec = {}
    rec = {}
    f_score = {}

    prec_nc = {}
    rec_nc = {}
    f_score_nc = {}

    for mn in models_names:

        test_acc[mn] = 100.0 * (tp[mn] + tn[mn]) / tot_count[mn]

        for i in range(n):
            class_acc[mn][i] = 100.0 * class_correct[mn][i] / max(class_total[mn][i], 1e-5)
            class_acc[mn][i] = float(class_acc[mn][i])

        model_tp = tp[mn]
        model_tn = tn[mn]
        model_fp = fp[mn]
        model_fn = fn[mn]

        prec[mn] = model_tp / (model_tp + model_fp)
        rec[mn] = model_tp / (model_tp + model_fn)
        f_score[mn] = 2.0 * (prec[mn] * rec[mn]) / (prec[mn] + rec[mn])

        prec_nc[mn] = model_tn / (model_tn + model_fn)
        rec_nc[mn] = model_tn / (model_tn + model_fp)
        f_score_nc[mn] = 2.0 * (prec_nc[mn] * rec_nc[mn]) / (prec_nc[mn] + rec_nc[mn])

    metrics_results = {
        "accuracy": test_acc,
        "class_accuracy": class_acc,
        "change precision": prec,
        "change recall": rec,
        "change f1-score": f_score,
        "no change precision": prec_nc,
        "no change recall": rec_nc,
        "no change f1-score": f_score_nc
    }

    return metrics_results
