import torch
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from lightly.data import LightlyDataset, SimCLRCollateFunction
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.helper_functions import yaml_loader, create_train_transforms
from tools.dataset import LightlyDatasetWithMasks
from tools.simple_clr import SimCLRModel


parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, help='Input size of image', default=128)
parser.add_argument('--save_top_k', type=int, help='Number of best checkpoints to save', default=3)
parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--num_workers', type=int, help='Number of workers', default=8)
parser.add_argument('--max_epochs', type=int, help='Number of epochs to train', default=100)
parser.add_argument('--log_steps', type=int, help='Length of logging interval', default=5)
parser.add_argument('--use_masks', type=bool, help='Whether to use segmentation masks', default=True)
args = parser.parse_args()

general_dict = yaml_loader('configs/general.yaml')

collate_fn = SimCLRCollateFunction(input_size=args.input_size, vf_prob=0.5, rr_prob=0.5)

if args.use_masks:
    dataset_train_simclr = LightlyDatasetWithMasks(
        input_dir=general_dict['path_to_train_data'],
        transform=create_train_transforms(resolution=args.input_size)
    )
else:
    dataset_train_simclr = LightlyDataset(
        input_dir=general_dict['path_to_train_data'],
        transform=create_train_transforms(resolution=args.input_size)
    )

checkpoint_callback = ModelCheckpoint(
    dirpath=general_dict['path_to_checkpoint'],
    monitor="train_loss_ssl",
    save_top_k=args.save_top_k,
    mode='min',
    filename='simclr_model_{train loss ssl:.3f}_{epoch:02d}',
    auto_insert_metric_name=True
)

dataloader_train_simclr = DataLoader(
    dataset_train_simclr,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=args.num_workers,
)

model = SimCLRModel()

logger = TensorBoardLogger("logs/tb_logs",
                           name="SimCLR",
                           default_hp_metric=False)
pl.utilities.distributed.log.setLevel(0)

trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    log_every_n_steps=args.log_steps,
    accelerator='cuda' if torch.cuda.is_available() else 'cpu',
    deterministic=True,
    devices=1,
    auto_lr_find=True,
    callbacks=checkpoint_callback,
    logger=logger,

)

if __name__ == '__main__':
    trainer.fit(model, dataloader_train_simclr)
