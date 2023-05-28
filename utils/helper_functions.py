import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image
from lightly.data import LightlyDataset, collate
from loguru import logger
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from torchvision import transforms

from tools.dataset import LightlyDatasetWithMasks
from tools.simple_clr import SimCLRModel


def yaml_loader(yaml_file):
    """
    Function to load yaml file.

    :param str yaml_file: path to yaml file
    :return: Dictionary of parameters
    :rtype: dict
    """
    with open(yaml_file) as f:
        yaml_dict = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return yaml_dict


def generate_embeddings(model, dataloader):
    """
    Generates representations for all images in the dataloader with
    the given model.

    :param simple_clr.SimCLRModel model: trained CLR model
    :param torch.utils.data.DataLoader dataloader: dataloader of images
    :returns: embeddings of images, filenames of images
    :rtype: torch.Tensor, str
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, label, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


def get_image_as_np_array(filename: Union[str, Path]):
    """
    Returns an image as a numpy array.

    :param str filename: file name of an image
    :returns: numpy array of an image
    :rtype: np.array
    """
    img = Image.open(filename)
    return np.asarray(img)


def create_train_transforms(
        resolution,
        random_flip=True,
        color_jitter=False,
        random_rotation=True,
        normalize_image=True,
        blur=False,
        rot_degree=15,
        cj_brightness=0.4,
        cj_contrast=0.4,
        cj_saturation=0.4,
        cj_hue=0.1,
        blur_kernel_size=5,
        norm_mean=collate.imagenet_normalize['mean'],
        norm_std=collate.imagenet_normalize['std']):
    """
    Returns image transforms for training

    :param int resolution: size of image
    :param bool random_flip: whether random flip image or not
    :param bool color_jitter: whether jitter color of image or not
    :param bool random_rotation: whether random rotate image or not
    :param bool normalize_image: whether normalize image or not
    :param bool blur: whether blur image or not
    :param int rot_degree: rotation degree of image
    :param float cj_brightness: brightness of image
    :param float cj_contrast: contrast of image
    :param float cj_saturation: saturation of image
    :param float cj_hue: hue of image
    :param int blur_kernel_size: kernel size of blurring
    :param list norm_mean: list of means
    :param list norm_std: list of standard deviations
    :returns: compose of transforms of image
    :rtype: transforms.Compose
    """
    data_transforms = [transforms.Resize((resolution, resolution))]

    if random_flip:
        data_transforms.append(transforms.RandomHorizontalFlip())

    if color_jitter:
        data_transforms.append(transforms.ColorJitter(
            brightness=cj_brightness,
            contrast=cj_contrast,
            saturation=cj_saturation,
            hue=cj_hue
        ))

    if random_rotation:
        data_transforms.append(transforms.RandomRotation(degrees=rot_degree))

    if blur:
        data_transforms.append(transforms.GaussianBlur(kernel_size=blur_kernel_size))

    if normalize_image:
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    data_transforms.append(transforms.ToPILImage())

    return transforms.Compose(data_transforms)


def create_test_transforms(
        resolution,
        normalize_image=True,
        norm_mean=collate.imagenet_normalize['mean'],
        norm_std=collate.imagenet_normalize['std']):
    """
    Returns image transforms for testing

    :param int resolution: size of image
    :param bool normalize_image: whether normalize image or not
    :param list norm_mean: list of means
    :param list norm_std: list of standard deviations
    :returns: compose of transforms of image
    :rtype: transforms.Compose
    """
    data_transforms = [
        transforms.Resize((resolution, resolution)),
    ]

    if normalize_image:
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    # data_transforms.append(transforms.ToPILImage())

    return transforms.Compose(data_transforms)


def plot_knn_examples_for_uploaded_image(embeddings, filenames, path_to_test_data,
                                         query_filename, n_neighbors=3,
                                         save_dir=None):
    """
    Plots nearest neighbors of a specific image given its filename

    :param np.array embeddings: embeddings of images
    :param str filenames: file names of images
    :param str path_to_test_data: path to test data
    :param int query_filename: specified image file name
    :param int n_neighbors: number of nearest neighbors to plot
    :param str save_dir: path to save image with nearest images
    """
    # Get the index of the query image
    query_idx = filenames.index(str(query_filename))
    # Get the embedding of the query image
    query_embedding = embeddings[query_idx]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)

    # Compute neighbors for the query image
    distances, indices = nbrs.kneighbors([query_embedding])
    fig = plt.figure()

    for plot_x_offset, neighbor_idx in enumerate(indices[0]):
        ax = fig.add_subplot(1, n_neighbors, plot_x_offset + 1)
        fname = Path(path_to_test_data).joinpath(filenames[neighbor_idx])
        plt.imshow(get_image_as_np_array(fname))
        ax.set_title(f'd={distances[0][plot_x_offset]:.3f}')
        plt.axis('off')

    if save_dir:
        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir.joinpath(f'nearest_neighbors_{query_filename}.png'))



