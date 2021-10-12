import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from monai.transforms.spatial.dictionary import RandRotated
from monai.transforms.utility.dictionary import ConcatItemsd

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from monai.data import CacheDataset
from monai.data.dataset import PersistentDataset
from monai.data.utils import list_data_collate, partition_dataset
from monai.transforms import (AddChanneld, Compose, CropForegroundd,
                              EnsureTyped, LoadImaged, Orientationd,
                              RandCropByPosNegLabeld, RandRotate90d, Resized,
                              ScaleIntensityd, Spacingd)
from monai.transforms.croppad.dictionary import (CenterSpatialCropd,
                                                 RandSpatialCropSamplesd,
                                                 ResizeWithPadOrCropd)
from monai.transforms.intensity.dictionary import HistogramNormalized
from torch.utils.data import DataLoader
from tqdm import tqdm

import toolbox.constants as C
from toolbox.test_monai import DicomSeries3DReader

@dataclass
class MonaiDatasetConfig:
    sequences: List[str]
    dataset_dir: str

    train_val_split_ratio: float = 0.8
    seed: int = 777

    train_batch_size: int = 2
    num_workers: int = 4
    
    gen_train_transforms_func: Optional[Callable[[str], Compose]] = None  # input argument of the function is sequence type
    gen_val_transforms_func: Optional[Callable[[str], Compose]] = None  # input argument of the function is sequence type

    cache_dir: str = "./cache"


class RSNA_MICCAI_MONAI_Dataset(pl.LightningDataModule):
    def __init__(self, dataset_config: MonaiDatasetConfig):
        super().__init__()
        self.config = dataset_config

        self.source_train_data_list = []
        self.train_data_list = []
        self.val_data_list = []
        self.test_data_list = []

        self.train_transforms = self.gen_train_transforms()
        self.val_transforms = self.gen_val_transforms()
        self.test_transforms = self.gen_val_transforms()
        self.prediction_transforms = self.gen_val_transforms()

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.prediction_ds = None

    def _gen_subject_data_list_from_csv(self, mode: str) -> List[Dict[str, Union[str, int]]]:
        if mode not in ["train", "test"]:
            raise ValueError(f"Invalid mode {mode}")

        csv_file = "train_labels.csv" if mode == "train" else "sample_submission.csv"
        df = pd.read_csv(os.path.join(self.config.dataset_dir, csv_file), dtype={"BraTS21ID": str, "MGMT_value": int})
        subject_dict_list = []
        for _, row in df.iterrows():
            subject_id = row["BraTS21ID"]
            # skip subjects mentioned in
            # https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/262046
            if subject_id in ["00109", "00123", "00709"]:
                continue
            subject_dict_list.append({
                C.T1: os.path.join(self.config.dataset_dir, mode, subject_id, C.T1),
                C.T2: os.path.join(self.config.dataset_dir, mode, subject_id, C.T2),
                C.T1GD: os.path.join(self.config.dataset_dir, mode, subject_id, C.T1GD),
                C.FLAIR: os.path.join(self.config.dataset_dir, mode, subject_id, C.FLAIR),
                C.LABEL: row["MGMT_value"]
            })
        return subject_dict_list

    def gen_train_transforms(self):
        if callable(self.config.gen_train_transforms_func):
            return self.config.gen_train_transforms_func(self.config.sequences)
        else:
            sequences = self.config.sequences
            return Compose([
                LoadImaged(keys=sequences),
                AddChanneld(keys=sequences),
                Orientationd(keys=sequences, axcodes="RAS"),
                HistogramNormalized(keys=sequences),
                CropForegroundd(keys=sequences, source_key=sequences[0]),
                Spacingd(
                    keys=sequences,
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear"),
                ),
                # CenterSpatialCropd(keys=sequences, roi_size=(256,256,256)),
                # RandSpatialCropSamplesd(
                #     keys=sequences,
                #     roi_size=(64, 64, 64),
                #     num_samples=4
                # ),
                ResizeWithPadOrCropd(keys=sequences, spatial_size=(96,96,96), mode="minimum"),
                ScaleIntensityd(keys=sequences),
                # Resized(keys=sequences, spatial_size=(96, 96, 96)),
                RandRotated(keys=sequences, range_x=15, range_y=15, range_z=15,prob=0.8),
                # RandRotate90d(keys=sequences, prob=0.8, spatial_axes=[0, 2]),
                ConcatItemsd(keys=sequences, name="ALL_SEQ"),
                EnsureTyped(keys=["ALL_SEQ"]),
                # EnsureTyped(keys=[sequences]),
            ])
    def gen_val_transforms(self):
        if callable(self.config.gen_val_transforms_func):
            return self.config.gen_val_transforms_func(self.config.sequences)
        else:
            sequences = self.config.sequences
            return Compose([
                LoadImaged(keys=sequences),
                AddChanneld(keys=sequences),
                Orientationd(keys=sequences, axcodes="RAS"),
                HistogramNormalized(keys=sequences),
                CropForegroundd(keys=sequences, source_key=sequences[0]),
                Spacingd(
                    keys=sequences,
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear"),
                ),
                # CenterSpatialCropd(keys=sequences, roi_size=(256,256,256)),
                ResizeWithPadOrCropd(keys=sequences, spatial_size=(96,96,96), mode="minimum"),
                ScaleIntensityd(keys=sequences),
                # Resized(keys=sequences, spatial_size=(96, 96, 96)),
                ConcatItemsd(keys=sequences, name="ALL_SEQ"),
                EnsureTyped(keys=["ALL_SEQ"]),
                # EnsureTyped(keys=sequences),
            ])

    def prepare_data(self) -> None:
        self.source_train_data_list = self._gen_subject_data_list_from_csv("train")
        self.test_data_list = self._gen_subject_data_list_from_csv("test")

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_data_list, self.val_data_list = partition_dataset(
                self.source_train_data_list,
                ratios=[self.config.train_val_split_ratio, (1 - self.config.train_val_split_ratio)],
                shuffle=True,
                seed=self.config.seed,
            )


            self.train_ds = PersistentDataset(
                data=self.train_data_list,
                transform=self.train_transforms,
                # cache_rate=1.0, num_workers=4,
                cache_dir = self.config.cache_dir
                )
            self.val_ds = PersistentDataset(
                data=self.val_data_list,
                transform=self.val_transforms,
                # cache_rate=1.0, num_workers=4,
                cache_dir = self.config.cache_dir
                )
            # self.train_ds = CacheDataset(
            #     data=self.train_data_list,
            #     transform=self.train_transforms,
            #     cache_rate=1.0, num_workers=4,
            #     # cache_dir = "./cache"
            #     )
            # self.val_ds = CacheDataset(
            #     data=self.val_data_list,
            #     transform=self.val_transforms,
            #     cache_rate=1.0, num_workers=4,
            #     # cache_dir = "./cache"
            #     )

        if stage == "test" or stage is None:
            self.test_ds = CacheDataset(
                data=self.test_data_list,
                transform=self.val_transforms,
                cache_rate=1.0, num_workers=4,
                #cache_dir = "./cache"
                )

        
        

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.config.train_batch_size, shuffle=True,
            num_workers=self.config.num_workers, collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=1, num_workers=self.config.num_workers)
        return val_loader
    
    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_ds, batch_size=1, num_workers=self.config.num_workers)
        return test_loader

    def predict_dataloader(self):
        predict_loader = torch.utils.data.DataLoader(
            self.test_ds, batch_size=1, num_workers=self.config.num_workers)
        return predict_loader



