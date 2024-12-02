from typing import Callable
from monai.transforms.compose import Compose
from monai.transforms.post.array import Activations, AsDiscrete
from monai.transforms.utility.array import EnsureType
import pytorch_lightning as pl
import torch
import torchmetrics
import monai.metrics.metric
from monai.metrics.rocauc import ROCAUCMetric
from dataclasses import dataclass, field
import toolbox.constants as C
from monai.data.utils import decollate_batch


@dataclass
class ModelConfig:
    criterion: torch.nn.modules.Module = field(
        default_factory=torch.nn.CrossEntropyLoss
    )
    optimizer_class: Callable = torch.optim.AdamW
    accuracy: torchmetrics.Metric = field(default_factory=torchmetrics.Accuracy)
    auc: monai.metrics.metric.Metric = field(default_factory=ROCAUCMetric)
    lr: float = 1e-4
    weight_decay: float = 1e-5


class Model(pl.LightningModule):
    def __init__(self, sequence, net, config: ModelConfig) -> None:
        super().__init__()
        self.sequence = sequence
        self.net = net
        self.config = config
        self.post_pred = Compose([EnsureType(), Activations(softmax=True)])
        self.post_label = Compose(
            [EnsureType(), AsDiscrete(to_onehot=True, n_classes=2)]
        )
        self.lr = self.config.lr

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        """
        Even though model is initialized when passed to .fit() function, Calls to configure_optimizers function are
        lazy, which means that it won't actually be called untill it is needed for training.
        """
        optimizer = self.config.optimizer_class(
            self.net.parameters(), lr=self.lr, weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def prepare_batch(self, batch):
        return batch[self.sequence], batch[C.LABEL]

    def infer_batch(self, batch):
        """
        Perform inference over a batch and return predicted value along with
        the ground truth.
        """
        x, y = self.prepare_batch(batch)
        y_hat = self.forward(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.config.criterion(y_hat, y)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        preds = torch.argmax(y_hat, dim=1)
        acc = self.config.accuracy(preds.cpu(), y.cpu())
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.config.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = self.config.accuracy(preds.cpu(), y.cpu())

        preds = [self.post_pred(i) for i in decollate_batch(y_hat)]
        labels = [self.post_label(i) for i in decollate_batch(y)]
        self.config.auc(preds, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        auc_result = self.config.auc.aggregate()
        self.log("auc", auc_result, prog_bar=True, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Perform inference over the batch and return model prediction.
        """
        y, _ = self.infer_batch(batch)
        return torch.argmax(y, dim=1)
