import torch
import argparse
import pytorch_lightning as pl

from lightly.data import LightlyDataset, SimCLRCollateFunction

from utils.helper_functions import yaml_loader, create_train_transforms
from tools.simple_clr import SimCLRModel

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, help='Input size of image', default=128)
parser.add_argument('--save_top_k', type=int, help='Number of best checkpoints to save', default=5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--num_workers', type=int, help='Number of workers', default=8)
parser.add_argument('--max_epochs', type=int, help='Number of epochs to train', default=100)
parser.add_argument('--log_steps', type=int, help='Length of logging interval', default=5)
args = parser.parse_args()

general_dict = yaml_loader('configs/general.yaml')

collate_fn = SimCLRCollateFunction(input_size=args.input_size, vf_prob=0.5, rr_prob=0.5)

dataset_train_simclr = LightlyDataset(
    input_dir=general_dict['path_to_train_data'],
    transform=create_train_transforms(resolution=args.input_size)
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=general_dict['path_to_checkpoint'],
                                                   save_top_k=args.save_top_k, monitor="train_loss_ssl")

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=args.num_workers,
)

if __name__ == '__main__':
    model = SimCLRModel()
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_steps,
        devices=1,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dataloader_train_simclr)
