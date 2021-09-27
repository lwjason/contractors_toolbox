import os
import pandas as pd
import numpy as np
import torchio as tio
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from toolbox.constants import T1, T1GD, T2, FLAIR


class DatasetConfig:
    all_sequence = [T1, T1GD, T2, FLAIR]
    sequence = [FLAIR]  # accept multiple inputs e.g. [T1, FLAIR]
    train_test_split_method: str = "random"  # random / sklearn / custom
    # TorchIO resample template space
    working_space = T2  # T1 / T1GD / T2 / FLAIR
    # TorchIO Augmentation parameter
    RandomGamma: float = 0.5
    RandomNoise: float = 0.5
    RandomMotion: float = 0.1
    RandomBiasField: float = 0.25


class RSNA_MICCAIBrainTumorDataset(pl.LightningDataModule):
    """ Represents pytorch_lightning.LightningDataModule object that is specific to
    RSNA MICCAI-Brain Tumor Kaggle Competition. Although, it can easily be generalised
    to other BINARY CLASSIFICATION tasks.
    Usage: RSNA_MICCAIBrainTumorDataset(dataset_dir, batch_size, train_label_csv, train_val_ratio).setup()
    """

    def __init__(self, dataset_dir, batch_size, train_label_csv, train_val_ratio, task=None, preprocess=None,
                 augmentation=None):
        super().__init__()
        self.task = task
        self.batch_size = batch_size
        self.dataset_dir = Path(dataset_dir)
        self.train_val_ratio = train_val_ratio
        self.train_label_df = pd.read_csv(train_label_csv)
        self.augmentation = augmentation
        self.preprocess = preprocess
        self.subjects = None
        self.test_subjects = None
        self.transforms = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    @staticmethod
    def get_preprocess_transform(preprocess):
        """ If specified, returns sequence of preprocessing functions. Otherwise,
        default sequence is returned.
        Default space to be resampled is T2w
        :param preprocess - sequence of custom preprocessing techniques
        :return sequence of custom, or default preprocessing techniques
        """
        if not preprocess:
            return tio.Compose(
                [
                    tio.ToCanonical(),
                    tio.Resample(DatasetConfig.working_space),
                    # tio.RescaleIntensity((-1, 1)),
                    tio.OneHot(),
                ]
            )
        return preprocess

    @staticmethod
    def get_augmentation_transform(augment):
        """ If specified, returns sequence of online augmentation functions. Otherwise,
        default sequence is returned.
        :param augment - sequence of custom augmentation techniques
        :return sequence of custom, or default augmentation techniques
        """

        if augment:
            return augment

        else:
            return tio.Compose([
                tio.RandomAffine(),
                tio.RandomGamma(p=DatasetConfig.RandomGamma),
                tio.RandomNoise(p=DatasetConfig.RandomNoise),
                tio.RandomMotion(p=DatasetConfig.RandomMotion),
                tio.RandomBiasField(p=DatasetConfig.RandomBiasField),
            ])

    @staticmethod
    def get_max_shape(subjects):
        """ Returns maximum shape based on list of subjects
        :param subjects - list of subjects for which we want
         to retrieve maximum shape
        :return maximum subject shape
        """
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def split_data(self, data_split, splitting_func, kwargs):
        """Create train and validation data"""
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects

        if data_split == "random":
            from torch.utils.data import random_split
            train_subjects, val_subjects = random_split(self.subjects, splits)
        elif data_split == "sklearn":
            from sklearn.model_selection import train_test_split
            train_subjects, val_subjects = train_test_split(self.subjects, test_size=self.train_val_ratio)
        elif data_split == "custom" and splitting_func and kwargs:
            train_subjects, val_subjects = splitting_func(**kwargs)
        else:
            raise ValueError(
                "[ERROR]: Please specify one of the following splitting criteria: random, sklearn or custom")
        return train_subjects, val_subjects

    def prepare_data(self, sequence):
        """ Creates a list of tio.Subject(s) for both, training and test sets.
        by default its FLAIR, but it could also be a LIST of sequences as well.
        """
        subject_train_dict, subject_test_dict = self.get_subject_dicts()
        subject_train_labels = self.get_subject_labels(subject_column="BraTS21ID", label_column="MGMT_value")
        self.subjects = self.create_subjects(sequence, subject_train_dict, subject_train_labels)
        self.test_subjects = self.create_subjects(sequence, subject_test_dict, None)


    def create_subjects(self, sequences, subject_dict, subject_labels):
        subjects = []
        for subject_id, sequence_dict in subject_dict.items():
            tio_subject_dict = {
                "subject_id": subject_id,
            }
            if subject_labels:
                subject_label = subject_labels[subject_id]
                tio_subject_dict["label"] = subject_label
            for sequence in sequences:
                tio_subject_dict[sequence] = tio.ScalarImage(sequence_dict[sequence])

            subjects.append(tio.Subject(tio_subject_dict))
        return subjects

    def get_subject_labels(self, subject_column, label_column, zfill=5):
        """ Creates subject_id : label dictionary
        :param subject_column - column with subject ID
        :param label_column - MGMT value
        :param zfill=5 - for RSNA MICCAI-Brain Tumor competition subject ID's
         are represented as strings with up to 5 leading zeros
        """
        return {
            str(row[subject_column]).zfill(zfill): row[label_column]
            for idx, row in self.train_label_df.iterrows()
        }

    def get_subject_dicts(self):
        """ Returns dictionary where key represents subject directory path, and value
        represents path to DICOM sequence.

        return {
            subject_id: {
                sequence: sequence_path
            }
        }

        """
        train_dict, test_dict = {}, {}
        for (dataset_type, subject_dict) in [('train', train_dict), ('test', test_dict)]:
            dataset_path = os.path.join(self.dataset_dir, dataset_type)
            for subject_dir in os.listdir(dataset_path):
                subject_dir_path = os.path.join(dataset_path, subject_dir)
                for sequence_dir in os.listdir(subject_dir_path):
                    sequence_path = os.path.join(subject_dir_path, sequence_dir)
                    try:
                        subject_dict[subject_dir][sequence_dir] = sequence_path
                    except KeyError:
                        subject_dict[subject_dir] = {sequence_dir: sequence_path}

        return train_dict, test_dict

    def train_dataloader(self):
        """ Returns PyTorch DataLoader for TRAINING set.
        """
        return DataLoader(self.train_set, self.batch_size)

    def val_dataloader(self):
        """ Returns PyTorch DataLoader for VALIDATION set.
        """
        return DataLoader(self.val_set, self.batch_size)

    def test_dataloader(self):
        """ Returns PyTorch DataLoader for TEST set.
        """
        return DataLoader(self.test_set, self.batch_size)

    def setup(self,
              data_split=DatasetConfig.train_test_split_method,
              splitting_func=None,
              kwargs=None,
              stage=None
              ):
        """ Set's up TRAINING, VALIDATION and TEST datasets according to the specified
        splitting criteria.
        :param data_split - how the data is going to be split, by default its random, but
         other options include: sklearn and custom
        :param splitting_func - pointer to the custom splitting function if applicable
        :param kwargs - dictionary of parameters for the custom splitting function. Must
         be passed together with splitting_func parameter
        """
        preprocess = self.get_preprocess_transform(self.preprocess)
        augmentation = self.get_augmentation_transform(self.augmentation)
        self.transforms = tio.Compose([preprocess, augmentation])

        self.prepare_data()
        train_subjects, val_subjects = self.split_data(data_split, splitting_func, kwargs)
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transforms)
        # val and test dataset should not apply augmentation methods.
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)
