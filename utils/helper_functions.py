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
    with open(yaml_file) as f:
        yaml_dict = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return yaml_dict


def generate_embeddings(model, dataloader):
    """
    Generates representations for all images in the dataloader with
    the given model
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
    Returns an image as an numpy array
    """
    img = Image.open(filename)
    return np.asarray(img)


def create_train_transforms(
    resolution: int,
    random_flip: bool = True,
    color_jitter: bool = False,
    random_rotation: bool = True,
    normalize: bool = True,
    blur: bool = False,
    rot_degree: int = 15,
    cj_brightness: float = 0.4,
    cj_contrast: float = 0.4,
    cj_saturation: float = 0.4,
    cj_hue: float = 0.1,
    blur_kernel_size: int = 5,
    norm_mean: list = collate.imagenet_normalize['mean'],
    norm_std: list = collate.imagenet_normalize['std'],
) -> transforms.Compose:
    """
    Returns image transforms for training
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
    resolution: int,
    normalize: bool = True,
    norm_mean: list = collate.imagenet_normalize['mean'],
    norm_std: list = collate.imagenet_normalize['std'],
) -> transforms.Compose:
    """
    Returns image transforms for testing
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
    Plots multiple rows of random images with their nearest neighbors
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
        plt.savefig(os.path.join(save_path, 'examples.png'))

