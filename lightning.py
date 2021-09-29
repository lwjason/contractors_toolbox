import os
import monai
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torchio as tio
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from toolbox.dataset import RSNA_MICCAIBrainTumorDataset, DatasetConfig
from toolbox.dicom import DicomReader
from toolbox.constants import *

from tqdm import tqdm
from playground import pytorch_monai as pm
from torchio.transforms import HistogramStandardization
from pathlib import Path

class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch['FLAIR'][tio.DATA], batch['label']

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


def main():
    landmarks_dict = {
        T1: Path(".") / "histogram_landmarks" / "t1w_landmarks.npy",
        T2: Path(".") / "histogram_landmarks" / "t2w_landmarks.npy",
        FLAIR: Path(".") / "histogram_landmarks" / "flair_landmarks.npy",
        T1GD: Path(".") / "histogram_landmarks" / "t1wce_landmarks.npy",
    }

    preprocess = tio.Compose([
        tio.RescaleIntensity((-1, 1)),
        tio.ToCanonical(),
        tio.Resample((1.0, 1.0, 1.0)),
        tio.Resize((256, 256, 256)),
        tio.HistogramStandardization(landmarks_dict)
    ])
    dataset = RSNA_MICCAIBrainTumorDataset(
        dataset_dir="/kaggle/input",
        batch_size=1,
        train_label_csv="/kaggle/input/train_labels.csv",
        train_val_ratio=0.8,
        preprocess=preprocess
    )
    dataset.setup()
    print(f"[INFO]: Number of Subjects in TRAINING Set: {len(dataset.train_set)}")
    print(f"[INFO]: Number of Subjects in VALIDATION Set: {len(dataset.val_set)}")
    print(f"[INFO]: Number of Subjects in TEST Set: {len(dataset.test_set)}\n")

    densenet = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)

    model = Model(net=densenet,
                  criterion=torch.nn.CrossEntropyLoss(),
                  optimizer_class=torch.optim.AdamW,
                  learning_rate=1e-2
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