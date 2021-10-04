from pathlib import Path
from typing import List

import monai
import pytorch_lightning as pl
import torch
import torchio as tio
import pandas as pd

from toolbox.lightning import Model
from toolbox.constants import *
from toolbox.dataset import RSNA_MICCAIBrainTumorDataset


def get_preprocess(landmarks_dict):
    preprocess = tio.Compose(
        [
            tio.HistogramStandardization(landmarks_dict),
            tio.RescaleIntensity((-1, 1)),
            tio.ToCanonical(),
            tio.Resample((1.0, 1.0, 1.0)),
            tio.Resize((256, 256, 256)),
        ]
    )
    return preprocess


def get_model(net, checkpoint_path: str):
    model = Model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        net=net,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.AdamW,
        learning_rate=1e-2,
    )
    return model


def save_submission_csv(results: List[torch.Tensor], dataset: RSNA_MICCAIBrainTumorDataset, csv_name="submission.csv"):
    predictions = []
    for r, subject in zip(results, dataset.test_set.dry_iter()):
        prediction = {
            "BraTS21ID": subject["subject_id"],
            "MGMT_value": r.cpu().numpy()[0]
        }
        predictions.append(prediction)
    df = pd.DataFrame(predictions)
    df.to_csv(csv_name, index=False)


def predict_and_submit(model: pl.LightningModule, dataset: pl.LightningDataModule, gpus=0):
    trainer = pl.Trainer(gpus=gpus)
    results = trainer.predict(model=model, datamodule=dataset)

    save_submission_csv(results, dataset)


if __name__ == "__main__":
    landmarks_dict = {
        T1: torch.load(Path(".") / "histogram_landmarks" / "t1w_landmarks.npy"),
        T2: torch.load(Path(".") / "histogram_landmarks" / "t2w_landmarks.npy"),
        FLAIR: torch.load(Path(".") / "histogram_landmarks" / "flair_landmarks.npy"),
        T1GD: torch.load(Path(".") / "histogram_landmarks" / "t1wce_landmarks.npy"),
    }
    preprocess = tio.Compose(
        [
            tio.HistogramStandardization(landmarks_dict),
            tio.RescaleIntensity((-1, 1)),
            tio.ToCanonical(),
            tio.Resample((1.0, 1.0, 1.0)),
            tio.Resize((256, 256, 256)),
        ]
    )

    dataset = RSNA_MICCAIBrainTumorDataset(
        dataset_dir="/kaggle/input",
        batch_size=1,
        train_label_csv="/kaggle/input/train_labels.csv",
        train_val_ratio=0.8,
        preprocess=preprocess,
    )
    dataset.setup()

    checkpoint_path = "lightning_logs/version_4/checkpoints/epoch=0-step=467.ckpt"
    
    densenet = monai.networks.nets.DenseNet121(
        spatial_dims=3, in_channels=1, out_channels=2
    )

    model = Model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        net=densenet,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.AdamW,
        learning_rate=1e-2,
    )

    predict_and_submit(model, dataset)
