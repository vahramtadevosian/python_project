import os
import cv2
import numpy as np

from PIL import Image
from lightly.data import LightlyDataset
from torchvision import transforms


class LightlyDatasetWithMasks(LightlyDataset):
    """
    A custom dataset class to add functionality to load
    and apply masks to images from the dataset.

    :param str input_dir: Path to images
    :param str mask_dir: Path to masks
    :param torchvision.transforms.Transform transform: Transforms applied to images
    """
    def __init__(self, input_dir, mask_dir, transform=None, test_mode=False):
        super().__init__(input_dir=input_dir, transform=transform)
        self.mask_dir = mask_dir
        self.test_mode = test_mode

    def __getitem__(self, index):
        image, target, filename = super().__getitem__(index)
        img_np = np.array(image)

        if self.test_mode:
            img_np = np.swapaxes(img_np, 0, 1)
            img_np = np.swapaxes(img_np, 1, 2)
        
        h, w, _ = img_np.shape

        mask_filename = os.path.join(self.mask_dir, '{}.png'.format(filename.split('.')[0].zfill(5)))
        mask = cv2.imread(mask_filename, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if (len(mask.shape) == 3) and (mask.shape[-1] == 4):
            mask = mask[:,:,0]

        masked_image = Image.fromarray((img_np * mask[:, :, None]/255.).astype(np.uint8))

        if self.test_mode:
            masked_image = transforms.ToTensor()(masked_image)

        return masked_image, target, filename