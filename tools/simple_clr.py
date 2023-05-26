from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim import SGD
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = SGD(self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optim, mode='min', patience=5)
        # scheduler = CosineAnnealingLR(optim, 10)  # doesn't work with pl
        return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "train_loss_ssl"}
