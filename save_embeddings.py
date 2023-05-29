import argparse
from pathlib import Path

import torch
from utils.helper_functions import yaml_loader, infer

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int,
                    help='Input size of image', default=256)
parser.add_argument('--batch_size', type=int,
                    help='Batch size', default=64)
parser.add_argument('--num_workers', type=int,
                    help='Number of workers', default=8)
args = parser.parse_args()

general_dict = yaml_loader('configs/general.yaml')


if __name__ == '__main__':
    for use_masks in [True, False]:
        embeddings, filenames = infer(use_masks, args, **general_dict)

        folder = Path('embeddings').joinpath(f'masked_{int(use_masks)}')

        torch.save(embeddings, folder.joinpath('embeddings.pt'))
        with open(folder.joinpath("filenames.txt"), 'w') as output:
            for row in filenames:
                output.write(str(row) + '\n')
