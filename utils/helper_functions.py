import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from lightly.data import collate


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


def get_image_as_np_array(filename: str):
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
        normalize=True,
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
    :param bool normalize: whether normalize image or not
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

    if normalize:
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    data_transforms.append(transforms.ToPILImage())

    return transforms.Compose(data_transforms)


def create_test_transforms(
        resolution,
        normalize=True,
        norm_mean=collate.imagenet_normalize['mean'],
        norm_std=collate.imagenet_normalize['std']):
    """
    Returns image transforms for testing

    :param int resolution: size of image
    :param bool normalize: whether normalize image or not
    :param list norm_mean: list of means
    :param list norm_std: list of standard deviations
    :returns: compose of transforms of image
    :rtype: transforms.Compose
    """
    data_transforms = [
        transforms.Resize((resolution, resolution)),
    ]

    if normalize:
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    # data_transforms.append(transforms.ToPILImage())

    return transforms.Compose(data_transforms)


def plot_knn_examples(embeddings, filenames, path_to_test_data, n_neighbors=3, num_examples=6, save_path=None):
    """
    Plots multiple rows of random images with their nearest neighbors.

    :param np.array embeddings: embeddings of images
    :param str filenames: file names of images
    :param str path_to_test_data: path to test data
    :param int n_neighbors: number of nearest neighbors to plot
    :param int num_examples: number of examples
    :param str save_path: path to save image with nearest images
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    for idx in samples_idx:
        fig = plt.figure()
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            fname = os.path.join(path_to_test_data, filenames[neighbor_idx])
            plt.imshow(get_image_as_np_array(fname))
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            plt.axis("off")

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f'examples_{idx}.png'))


def plot_knn_examples_for_uploaded_image(embeddings, filenames, path_to_test_data, query_filename, n_neighbors=3,
                                         save_path=None):
    """
    Plots nearest neighbors of a specific image given its filename

    :param np.array embeddings: embeddings of images
    :param str filenames: file names of images
    :param str path_to_test_data: path to test data
    :param int query_filename: specified image file name
    :param int n_neighbors: number of nearest neighbors to plot
    :param str save_path: path to save image with nearest images
    """
    query_idx = filenames.index(query_filename)  # Get the index of the query image
    query_embedding = embeddings[query_idx]  # Get the embedding of the query image
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors([query_embedding])  # Compute neighbors for the query image
    fig = plt.figure()
    for plot_x_offset, neighbor_idx in enumerate(indices[0]):
        ax = fig.add_subplot(1, n_neighbors, plot_x_offset + 1)
        fname = os.path.join(path_to_test_data, filenames[neighbor_idx])
        plt.imshow(get_image_as_np_array(fname))
        ax.set_title(f'd={distances[0][plot_x_offset]:.3f}')
        plt.axis('off')
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'nearest_neighbors_{query_filename}.png'))
