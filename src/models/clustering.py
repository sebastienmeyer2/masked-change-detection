"""Model based on PCA and clustering."""


from collections import Counter

import warnings

import numpy as np

import torch

import cv2
import skimage.morphology

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


from utils.data import apply_fixed_thresholding


def find_vector_set(img_diff, new_size):

    i = 0
    j = 0

    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))

    while i < vector_set.shape[0]:
        while j < new_size[1]:
            k = 0
            while k < new_size[0]:
                block = img_diff[j:j+5, k:k+5]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1

    mean_vec = np.mean(vector_set, axis=0)

    # Mean normalization
    vector_set = vector_set - mean_vec

    return vector_set, mean_vec


def find_fvs(evs, img_diff, mean_vec, new):

    i = 2
    feature_vector_set = []

    while i < new[1] - 2:
        j = 2
        while j < new[0] - 2:
            block = img_diff[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1

    fvs = np.dot(feature_vector_set, evs)
    fvs = fvs - mean_vec

    return fvs


def clustering(fvs, components, new):

    kmeans = KMeans(components, n_init="auto", verbose=0)
    kmeans.fit(fvs)
    output = kmeans.predict(fvs)
    count = Counter(output)

    least_index = min(count, key=count.get)
    change_map = np.reshape(output, (new[1] - 4, new[0] - 4))

    return least_index, change_map


def run_clustering(img_1, img_2, img_height=224, img_width=224, components=2, threshold=100):

    # Convert to arrays if needed
    if isinstance(img_1, torch.Tensor):

        img_1 = img_1.squeeze().detach().cpu().numpy()
        img_2 = img_2.squeeze().detach().cpu().numpy()

    # Transpose if needed
    if img_1.shape[0] <= 13:  # channels first

        img_1 = img_1.transpose((1, 2, 0))
        img_2 = img_2.transpose((1, 2, 0))

    # Resize Images
    new_size = np.asarray(img_1.shape) / 5
    new_size = new_size.astype(int) * 5
    img_1 = cv2.resize(img_1, (new_size[0], new_size[1])).astype(int)
    img_2 = cv2.resize(img_2, (new_size[0], new_size[1])).astype(int)

    # Difference Image
    img_diff = abs(img_1 - img_2)[:, :, 1]

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", category=RuntimeWarning)

        pca = PCA()
        vector_set, mean_vec = find_vector_set(img_diff, new_size)
        pca.fit(vector_set)
        evs = pca.components_

    fvs = find_fvs(evs, img_diff, mean_vec, new_size)

    # Compute change map
    least_index, change_map = clustering(fvs, components, new_size)

    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    change_map = change_map.astype(np.uint8)

    change_map_resize = cv2.resize(change_map, (img_height, img_width))

    # Closing / opening
    kernel = skimage.morphology.disk(6)

    close_map = cv2.morphologyEx(change_map, cv2.MORPH_CLOSE, kernel)
    close_map_resize = cv2.resize(close_map, (img_height, img_width))

    open_map = cv2.morphologyEx(close_map, cv2.MORPH_OPEN, kernel)
    open_map_resize = cv2.resize(open_map, (img_height, img_width))

    # Apply thresholding
    change_map_bin = apply_fixed_thresholding(
        change_map_resize, threshold=threshold, fill_value=1
    )
    close_map_bin = apply_fixed_thresholding(
        close_map_resize, threshold=threshold, fill_value=1
    )
    open_map_bin = apply_fixed_thresholding(
        open_map_resize, threshold=threshold, fill_value=1
    )

    return change_map_bin, close_map_bin, open_map_bin
