import os
import cv2
import numpy as np

from PIL import Image
from lightly.data import LightlyDataset


class LightlyDatasetWithMasks(LightlyDataset):
    """
    A custom dataset class to add functionality to load
    and apply masks to images from the dataset.

    :param str input_dir: Path to images
    :param str mask_dir: Path to masks
    :param torchvision.transforms.Transform transform: Transforms applied to images
    """
    def __init__(self, input_dir, mask_dir, transform=None):
        super().__init__(input_dir=input_dir, transform=transform)
        self.mask_dir = mask_dir

    def __getitem__(self, index):
        image, target, filename = super().__getitem__(index)
        img_np = np.array(image)
        h, w, _ = img_np.shape

        mask_filename = os.path.join(self.mask_dir, '{}.png'.format(filename.split('.')[0].zfill(5)))
        mask = cv2.imread(mask_filename, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        masked_image = Image.fromarray(img_np * mask[:, :, None])

        return masked_image, target, filename
