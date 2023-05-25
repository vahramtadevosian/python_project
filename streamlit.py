import torch
import argparse

from lightly.data import LightlyDataset

from utils.helper_functions import yaml_loader, generate_embeddings, plot_knn_examples, create_test_transforms

from tools.simple_clr import SimCLRModel


parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, help='Input size of image', default=128)
parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--num_workers', type=int, help='Number of workers', default=8)
args = parser.parse_args()

general_dict = yaml_loader('configs/general.yaml')

dataset_test = LightlyDataset(
    input_dir=general_dict['path_to_test_data'],
    transform=create_test_transforms(resolution=args.input_size)
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
)

if __name__ == '__main__':
    model = SimCLRModel.load_from_checkpoint(general_dict['checkpoint_path'])
    model.eval()
    embeddings, filenames = generate_embeddings(model.cpu(), dataloader_test)
    plot_knn_examples(embeddings, filenames, general_dict['path_to_test_data'], save_path=general_dict['plt_figures'])
