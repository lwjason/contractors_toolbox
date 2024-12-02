import os
from typing import Union, Sequence, Any, Tuple, Dict

import SimpleITK as sitk
import numpy as np
import pandas as pd
from monai.data import ImageReader

from .constants import *

LABEL = "label"


def gen_data_dicts(mode):
    """
    Generate the data dictionary list that MONAI prefers
    :param mode: can be train or test
    :return:list of data
    """
    if mode not in ["train", "test"]:
        raise ValueError(f"Invalid mode {mode}")

    data_list = []
    csv_file = "train_labels.csv" if "train" else "sample_submission.csv"

    df = pd.read_csv(os.path.join(INPUT_DIR, csv_file),
                     dtype={"BraTS21ID": str, "MGMT_value": int})

    for _, row in df.iterrows():
        subject_id = row["BraTS21ID"]
        # skip subjects mentioned in
        # https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/262046
        if subject_id in ["00109", "00123", "00709"]:
            continue
        data_list.append({
            T1: os.path.join(INPUT_DIR, mode, subject_id, T1),
            T2: os.path.join(INPUT_DIR, mode, subject_id, T2),
            T1GD: os.path.join(INPUT_DIR, mode, subject_id, T1GD),
            FLAIR: os.path.join(INPUT_DIR, mode, subject_id, FLAIR),
            LABEL: row["MGMT_value"] if "train" else None
        })
    return data_list


class DicomSeries3DReader(ImageReader):
    """
    Load the Dicom Series as a 3D array.
    It uses SimpleITK to read the Dicom Series.
    """
    def __init__(self) -> None:
        super().__init__()
        self.header = {}

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        """
        Implement ImageReader required abstract function.
        It's idea is to check whether the filename(here actually is a folder) is supported or not.
        :param filename: the folder path of the dicom series
        :return: True or False
        """
        return os.path.exists(filename) and os.path.isdir(filename)

    def read(self, data: Union[Sequence[str], str], **kwargs) -> Union[Sequence[Any], Any]:
        """
        Implement ImageReader required abstract function.
        Read the image and return the image object
        :param data: the folder path of the dicom series
        :return: simple itk image object
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(data)
        reader.SetFileNames(dicom_names)
        itk_image = reader.Execute()
        return itk_image

    def _prep_meta_dict(self, img: sitk.Image):
        """
        Get extra metadata that SimpleITK offers.
        :param img: simple itk image object
        """
        for k in img.GetMetaDataKeys():
            self.header[k] = img.GetMetaData(k)
        self.header["origin"] = np.asarray(img.GetOrigin())
        self.header["spacing"] = np.asarray(img.GetSpacing())
        self.header["direction"] = np.asarray([
            np.asarray(img.GetDirection()[:3]),
            np.asarray(img.GetDirection()[3:6]),
            np.asarray(img.GetDirection()[6:])])

    def _prep_affine(self, ):
        affine: np.ndarray = np.eye(self.header["direction"].shape[0] + 1)
        affine[(slice(-1), slice(-1))] = self.header["direction"] @ np.diag(self.header["spacing"])
        affine[(slice(-1), -1)] = self.header["origin"]
        self.header["original_affine"] = affine
        self.header["affine"] = self.header["original_affine"].copy()

    @staticmethod
    def _prep_spatial_shape(img: sitk.Image):
        shape = list(img.GetSize())
        shape.reverse()
        return np.asarray(shape)

    def _prep_header(self, img):
        """
        Prepare the header. Header is the bridge to other monai transforms.
        From my understanding, below headers are necessary.
        :param img: sitk image object
        """
        self._prep_meta_dict(img)
        self._prep_affine()
        self._prep_spatial_shape(img)
        self.header["original_channel_dim"] = "no_channel"

    def get_data(self, img: sitk.Image) -> Tuple[np.ndarray, Dict]:
        """
        Implement ImageReader required abstract function.
        Prepare the final returning objects
        :param img: sitk image object
        :return: the numpy array and header dict
        """
        self._prep_header(img)
        img_array = sitk.GetArrayViewFromImage(img)
        return img_array, self.header
