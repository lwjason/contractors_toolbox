import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
from random import sample
from pathlib import Path
from tqdm.notebook import tqdm_notebook

import torch
import torchio as tio
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from toolbox.dicom import DicomReader


class RSNA_MICCAIBrainTumorDataset(pl.LightningDataModule):
    """ Represents pytorch_lightning.LightningDataModule object that is specific to
    RSNA MICCAI-Brain Tumor Kaggle Competition. Although, it can easily be generalised
    to other BINARY CLASSIFICATION tasks.
    """

    def __init__(self, dataset_dir, batch_size, train_label_csv, train_val_ratio, task=None, preprocess=None,
                 augmentation=None):
        super().__init__()
        self.task = task
        self.batch_size = batch_size
        self.dataset_dir = Path(dataset_dir)
        self.train_val_ratio = train_val_ratio
        self.train_label_df = pd.read_csv(train_label_csv)
        self.subjects, self.test_subjects = [], []
        self.augmentation = augmentation
        self.preprocess = preprocess

        self.transforms = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self,
              data_split="random",
              splitting_func=None,
              kwargs=None,
              stage=None):
        """ Set's up TRAINING, VALIDATION and TEST datasets according to the specified
        sppliting criteria.

        :param data_split - how the data is going to be split, by default its random, but
         other options include: sklearn and custom
        :param splitting_func - pointer to the custom splitting function if applicable
        :param kwargs - dictionary of paramters for the custom splitting function. Must
         be passed together with splitting_func parameter
        """
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects

        splits = num_train_subjects, num_val_subjects
        if data_split == "random":
            train_subjects, val_subjects = random_split(self.subjects, splits)
        elif data_split == "sklearn":
            train_subjects, val_subjects = train_test_split(self.subjects, test_size=self.train_val_ratio)
        elif data_split == "custom" and splitting_func and kwargs:
            train_subjects, val_subjects = splitting_func(**kwargs)
        else:
            raise ValueError(
                "[ERROR]: Please specify one of the following splitting criteries: random, sklearn or custom")

        self.preprocess = self.get_preprocessing_transform(self.preprocess)
        self.augmentation = self.get_augmentation_transform(self.transforms)
        self.transforms = tio.Compose([self.preprocess, self.augmentation])

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transforms)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def prepare_data(self, sequence='FLAIR'):
        """ Creates a list of tio.Subject(s) for both, training and test sets.

        :param sequence: sequence for which training and test set should be configured.
        by default its FLAIR, but it could also be a LIST of sequences as well.
        """
        subject_train_dict, subject_test_dict = self.get_subject_dicts()
        subject_train_labels = self.get_subject_labels(subject_column="BraTS21ID", label_column="MGMT_value")

        for i, subject_id in zip(tqdm_notebook(range(len(subject_train_dict)), desc="Preparing Training Dataset"),
                                 subject_train_dict):
            subject_label = subject_train_labels[subject_id]
            if not isinstance(sequence, list):
                subject = self.create_tio_subject_single_image(subject_id=subject_id,
                                                               dicom_sequence_path=subject_train_dict[subject_id][
                                                                   sequence],
                                                               image_name=sequence,
                                                               label=subject_label)
            else:
                subject = self.create_tio_subject_multiple_images(subject_id=subject_id,
                                                                  dicom_sequence_paths=[
                                                                      subject_train_dict[subject_id][s] for s in
                                                                      sequence],
                                                                  image_names=[s for s in sequence],
                                                                  label=subject_label)
            self.subjects.append(subject)

        for i, subject_id in zip(tqdm_notebook(range(len(subject_test_dict)), desc="Preparing Test Dataset")
                , subject_test_dict):
            if not isinstance(sequence, list):
                subject = self.create_tio_subject_single_image(subject_id=subject_id,
                                                               dicom_sequence_path=subject_test_dict[subject_id][
                                                                   sequence],
                                                               image_name=sequence)
            else:
                subject = self.create_tio_subject_multiple_images(subject_id=subject_id,
                                                                  dicom_sequence_paths=[subject_test_dict[subject_id][s]
                                                                                        for s in sequence],
                                                                  image_names=[s for s in sequence])
            self.test_subjects.append(subject)

    def create_tio_subject_single_image(self, subject_id, dicom_sequence_path, image_name, label=None):
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

    def create_tio_subject_multiple_images(self, subject_id, dicom_sequence_paths, image_names, label=None):
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
        for idx in range(1, len(ALL_SEQUENCES)):
            tio_image = tio.ScalarImage(dicom_sequence_paths[idx])
            subject.add_image(image=tio_image,
                              image_name=image_names[idx])
        return subject

    def get_subject_labels(self, subject_column, label_column, zfill=5):
        """ Creates subject_id : label dictionary

        :param subject_colum - column with subject ID
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

    def get_max_shape(self, subjects):
        """ Returns maximum shape based on list of subjects

        :param subjects - list of subjects for which we want
         to retrieve maximum shape
        :retunr maximum subject shape
        """
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def get_preprocessing_transform(self, preprocess=None):
        """ If specified, returns sequence of preprocessing functions. Otherwise,
        default sequence is returned.

        :param preprocess - sequence of custom preprocessing techniques
        :return sequence of custom, or default preprocessing techniques
        """
        if not preprocess:
            return tio.Compose([
                tio.ToCanonical(),
                tio.Resample("T2w"),
                # tio.RescaleIntensity((-1, 1)),
                tio.OneHot(),
            ])
        return preprocess

    def get_augmentation_transform(self, augment=None):
        """ If specified, returns sequence of online augmentation functions. Otherwise,
        default sequence is returned.

        :param augment - sequence of custom augmentation techniques
        :return sequence of custom, or default augmentation techniques
        """
        if not augment:
            return tio.Compose([
                tio.RandomAffine(),
                tio.RandomGamma(p=0.5),
                tio.RandomNoise(p=0.5),
                tio.RandomMotion(p=0.1),
                tio.RandomBiasField(p=0.25),
            ])
        return augment

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

