import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import SGD
from lightly.loss import NTXentLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLRModel(pl.LightningModule):
    """
    Custom Contrastive Learning model with a ResNet-18 backbone
     and SimCLRProjectionHead projection head.
    """
    def __init__(self):
        super().__init__()

        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        """
        Forward pass of model.

        :param torch.Tensor x: input tensor for forward pass
        :returns: output of model after forward pass
        :rtype: torch.Tensor
        """
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch):
        """
        Training step of model.

        :param int batch: batch of images
        :returns: loss of a batch
        :rtype: float
        """
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        """
        Configures optimizer of model.

        :returns: dictionary of optimizer, learning scheduler and train loss
        :rtype: dict
        """
        optim = SGD(self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optim, mode='min', patience=5)
        return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "train_loss_ssl"}
