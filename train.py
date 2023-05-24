import torch
import pytorch_lightning as pl

from lightly.data import LightlyDataset, SimCLRCollateFunction

from utils.helper_functions import yaml_loader, create_train_transforms
from tools.simple_clr import SimCLRModel


general_dict = yaml_loader('configs/general.yaml')

collate_fn = SimCLRCollateFunction(input_size=general_dict['resolution'], vf_prob=0.5, rr_prob=0.5)

dataset_train_simclr = LightlyDataset(
    input_dir=general_dict['path_to_train_data'],
    transform=create_train_transforms(resolution=general_dict['resolution'])
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=general_dict['path_to_checkpoint'],
                                                   save_top_k=general_dict['save_top_k'], monitor="train_loss_ssl")

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=general_dict['batch_size'],
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=general_dict['num_workers'],
)

if __name__ == '__main__':
    model = SimCLRModel()
    trainer = pl.Trainer(
        max_epochs=general_dict['max_epochs'],
        log_every_n_steps=general_dict['log_steps'],
        devices=1,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dataloader_train_simclr)
