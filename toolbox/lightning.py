import monai
from datetime import datetime
import torch
import torchio as tio
import pytorch_lightning as pl
from toolbox.dataset import RSNA_MICCAIBrainTumorDataset
from toolbox.landmarks import landmarks_dict
from toolbox.constants import *


class Model(pl.LightningModule):
    def __init__(self, sequence, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.sequence = sequence
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch[self.sequence][tio.DATA], batch['label']

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        y, _ = self.infer_batch(batch)
        return y.argmax(dim=1)


def main():
    sequence = T1GD
    preprocess = tio.Compose([
        tio.HistogramStandardization(landmarks_dict),
        tio.RescaleIntensity((-1, 1)),
        tio.ToCanonical(),
        tio.Resample((1.0, 1.0, 1.0)),
        tio.Resize((256, 256, 256)),
    ])
    dataset = RSNA_MICCAIBrainTumorDataset(
        sequence=[sequence],
        dataset_dir="/kaggle/input",
        batch_size=1,
        train_label_csv="/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv/train_labels.csv",
        train_val_ratio=0.8,
        preprocess=preprocess
    )
    dataset.setup()
    print(f"[INFO]: Number of Subjects in TRAINING Set: {len(dataset.train_set)}")
    print(f"[INFO]: Number of Subjects in VALIDATION Set: {len(dataset.val_set)}")
    print(f"[INFO]: Number of Subjects in TEST Set: {len(dataset.test_set)}\n")

    densenet = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)

    model = Model(
        sequence= sequence,
        net=densenet,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.AdamW,
        learning_rate=5e-5,
    )

    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss'
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        callbacks=[early_stopping]
    )

    trainer.logger._default_hp_metric = False

    start = datetime.now()
    print("[INFO]: Training started at: ", start)
    trainer.fit(model=model, datamodule=dataset)
    print("[INFO]: Training duration: ", datetime.now() - start)


if __name__ == "__main__":
    main()