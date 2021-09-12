import os
from typing import Tuple, List
import pandas as pd
import numpy as np
import torchio as tio
import pytorch_lightning as pl
from pathlib import Path
from torchio import Subject
from tqdm.notebook import tqdm_notebook
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
        self.sequence = DatasetConfig.sequence
        self.augmentation = augmentation
        self.preprocess = preprocess

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
        if preprocess:
            return preprocess
        else:
            preprocess = tio.Compose(
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
    def create_tio_subject_single_image(subject_id, dicom_sequence_path, image_name, label=None) -> tio.Subject:
        """ Creates torchio.Subject and adds single image to it.
        :param subject_id: string which uniquely identifies subject id
        :param dicom_sequence_path: absolute path to DICOM sequence directory
        :param image_name: name of the image
        :param label: training label
        :returm instance of tio.Subject class
        """
        tio_image = tio.ScalarImage(dicom_sequence_path)
        subject = tio.Subject({
            "subject_id": subject_id,
            "label": label,
            image_name: tio_image
        })
        return subject

    @staticmethod
    def create_tio_subject_multiple_images(subject_id, dicom_sequence_paths, image_names, label=None):
        """ Creates torchio.Subject and adds sequence of images to it.
        :param subject_id: string which uniquely identifies subject id
        :param dicom_sequence_paths: absolute paths to DICOM sequence directories
        :param image_names: name of the images
        :param label: training label
        :returm instance of tio.Subject class
        """
        tio_image = tio.ScalarImage(dicom_sequence_paths[0])
        subject = tio.Subject({
            "subject_id": subject_id,
            "label": label,
            image_names[0]: tio_image
        })
        for idx in range(1, len(DatasetConfig.all_sequence)):
            tio_image = tio.ScalarImage(dicom_sequence_paths[idx])
            subject.add_image(image=tio_image,
                              image_name=image_names[idx])
        return subject

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

    def split_data(self, subjects, data_split, splitting_func, kwargs):
        """Create train and validation data"""
        num_subjects = len(subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects

        if data_split == "random":
            from torch.utils.data import random_split
            train_subjects, val_subjects = random_split(subjects, splits)
        elif data_split == "sklearn":
            from sklearn.model_selection import train_test_split
            train_subjects, val_subjects = train_test_split(subjects, test_size=self.train_val_ratio)
        elif data_split == "custom" and splitting_func and kwargs:
            train_subjects, val_subjects = splitting_func(**kwargs)
        else:
            raise ValueError(
                "[ERROR]: Please specify one of the following splitting criteria: random, sklearn or custom")
        return train_subjects, val_subjects

    def prepare_data(self) -> Tuple[List[Subject], List[Subject]]:
        """ Creates a list of tio.Subject(s) for both, training and test sets.
        by default its FLAIR, but it could also be a LIST of sequences as well.
        """
        subject_train_dict, subject_test_dict = self.get_subject_dicts()
        subject_train_labels = self.get_subject_labels(subject_column="BraTS21ID", label_column="MGMT_value")

        subjects = self.create_subjects(subject_train_dict, subject_train_labels)
        test_subjects = self.create_subjects(subject_test_dict)
        return subjects, test_subjects

    def create_subjects(self, subject_dict, subject_labels=None):
        subjects = []
        for i, subject_id in zip(tqdm_notebook(range(len(subject_dict)), desc="Preparing Training Dataset"),
                                 subject_dict):

            subject_label = subject_labels[subject_id] if subject_labels else None
            if len(self.sequence) == 1:
                sequence = self.sequence[0]
                subject = self.create_tio_subject_single_image(subject_id=subject_id,
                                                               dicom_sequence_path=subject_dict[subject_id][
                                                                   sequence],
                                                               image_name=sequence,
                                                               label=subject_label)
            else:
                subject = self.create_tio_subject_multiple_images(subject_id=subject_id,
                                                                  dicom_sequence_paths=[
                                                                      subject_dict[subject_id][s] for s in
                                                                      self.sequence],
                                                                  image_names=[s for s in self.sequence],
                                                                  label=subject_label)
            subjects.append(subject)
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

        subjects, test_subjects = self.prepare_data()
        train_subjects, val_subjects = self.split_data(subjects, data_split, splitting_func, kwargs)
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transforms)
        # val and test dataset should not apply augmentation methods.
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(test_subjects, transform=self.preprocess)