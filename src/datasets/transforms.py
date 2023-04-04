"""Transforms classes and functions."""


import random

import numpy as np
from PIL import ImageOps

import torch
from torchvision.transforms import Compose
from torchvision.transforms.functional import normalize, resize, rotate, to_tensor


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class RandomFlipPair(object):
    """Flip randomly the images in a sample."""
    def __init__(self, proba_flip=0.):

        self.proba_flip = proba_flip

    def __call__(self, sample):

        img_1, img_2, label = sample

        if self.proba_flip > 0 and random.random() < self.proba_flip:

            img_1 = img_1.numpy()[:, :, ::-1].copy()
            img_1 = torch.from_numpy(img_1)

            img_2 = img_2.numpy()[:, :, ::-1].copy()
            img_2 = torch.from_numpy(img_2)

            label = label.numpy()[:, ::-1].copy()
            label = torch.from_numpy(label)

        sample = (img_1, img_2, label)

        return sample


class RandomRotPair(object):
    """Rotate randomly the images in a sample per n * 90°."""
    def __init__(self, do_rotate=True):

        self.do_rotate = do_rotate

    def __call__(self, sample):

        img_1, img_2, label = sample

        if self.do_rotate:

            n = random.randint(0, 3)

            if n:

                img_1 = np.rot90(img_1.numpy(), n, axes=(1, 2)).copy()
                img_1 = torch.from_numpy(img_1)

                img_2 = np.rot90(img_2.numpy(), n, axes=(1, 2)).copy()
                img_2 = torch.from_numpy(img_2)

                label = np.rot90(label.numpy(), n, axes=(0, 1)).copy()
                label = torch.from_numpy(label)

        sample = (img_1, img_2, label)

        return sample


class RandomHorizontalFlipPair(object):
    """Horizontally flip the images in a sample."""
    def __init__(self, proba_flip=0):

        self.proba_flip = proba_flip

    def __call__(self, sample):

        img_1, img_2, label = sample

        if self.proba_flip > 0 and random.random() < self.proba_flip:

            img_1 = ImageOps.flip(img_1)
            img_2 = ImageOps.flip(img_2)
            label = ImageOps.flip(label)

        sample = (img_1, img_2, label)

        return sample


class RandomVerticalFlipPair(object):
    """Vertically flip the images in a sample."""
    def __init__(self, proba_flip=0):

        self.proba_flip = proba_flip

    def __call__(self, sample):

        img_1, img_2, label = sample

        if self.proba_flip > 0 and random.random() < self.proba_flip:

            img_1 = ImageOps.mirror(img_1)
            img_2 = ImageOps.mirror(img_2)
            label = ImageOps.mirror(label)

        sample = (img_1, img_2, label)

        return sample


class RandomRotationPair(object):
    """Rotate randomly the images in a sample."""
    def __init__(self, degree=0):

        self.degree = degree

    def __call__(self, sample):

        img_1, img_2, label = sample

        if self.degree > 0:

            img_1 = rotate(img_1, self.degree)
            img_2 = rotate(img_2, self.degree)
            label = rotate(label, self.degree)

        sample = (img_1, img_2, label)

        return sample


class ResizePair(object):
    """Resize images in a sample."""
    def __init__(self, nb_channels=3, img_height=224, img_width=224):

        self.nb_channels = nb_channels
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, sample):

        img_1, img_2, label = sample

        img_1 = resize(img_1, size=(self.img_height, self.img_width), antialias=True)
        img_2 = resize(img_2, size=(self.img_height, self.img_width), antialias=True)
        if isinstance(label, torch.Tensor) and label.ndim == 2:
            label = resize(
                label[None, :], size=(self.img_height, self.img_width), antialias=True
            ).squeeze()
        else:
            label = resize(label, size=(self.img_height, self.img_width), antialias=True)

        sample = (img_1, img_2, label)

        return sample


class ToTensorPair(object):
    """Convert to tensors images in a sample."""
    def __call__(self, sample):

        img_1, img_2, label = sample

        img_1 = to_tensor(img_1)
        img_2 = to_tensor(img_2)
        label = to_tensor(label).long().squeeze(0)

        sample = (img_1, img_2, label)

        return sample


class NormalizePair(object):
    """Normalize images in a sample."""
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):

        self.mean = mean
        self.std = std

    def __call__(self, sample):

        img_1, img_2, label = sample

        img_1 = normalize(img_1, mean=self.mean, std=self.std)
        img_2 = normalize(img_2, mean=self.mean, std=self.std)

        sample = (img_1, img_2, label)

        return sample


def create_change_detection_transform(
    proba_flip=0.,
    do_rotate=False
):

    transform = Compose(
        [
            RandomFlipPair(proba_flip=proba_flip),
            RandomRotPair(do_rotate=do_rotate),
        ]
    )

    return transform


def create_reconstruction_transform(
    size=224,
    degree=0,
    proba_hflip=0,
    proba_vflip=0,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD
):

    transform = Compose(
        [
            RandomHorizontalFlipPair(proba_flip=proba_hflip),
            RandomVerticalFlipPair(proba_flip=proba_vflip),
            RandomRotationPair(degree=degree),
            ResizePair(img_height=size, img_width=size),
            ToTensorPair(),
            NormalizePair(mean=mean, std=std)
        ]
    )

    return transform
