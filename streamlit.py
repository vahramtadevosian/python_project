import torch
import torchvision

from lightly.data import LightlyDataset

from utils.helper_functions import yaml_loader, generate_embeddings, plot_knn_examples, create_test_transforms

from tools.simple_clr import SimCLRModel


general_dict = yaml_loader('configs/general.yaml')

dataset_test = LightlyDataset(
    input_dir=general_dict['path_to_test_data'],
    transform=create_test_transforms(resolution=general_dict['resolution'])
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=general_dict['batch_size'],
    shuffle=False,
    drop_last=False,
    num_workers=general_dict['num_workers'],
)

if __name__ == '__main__':
    model = SimCLRModel.load_from_checkpoint(general_dict['checkpoint_path'])
    model.eval()
    embeddings, filenames = generate_embeddings(model.cpu(), dataloader_test)
    plot_knn_examples(embeddings, filenames, general_dict['path_to_test_data'], save_path='plt_figures')