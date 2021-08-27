import os
from typing import Union, Sequence, Any, Tuple, Dict

import SimpleITK as sitk
import numpy as np
from monai.data import ImageReader

from .constants import T1, T2, T1GD, FLAIR

LABEL = "label"


def gen_data_dicts(df, mode):
    """
    Generate the data dictionary list that MONAI prefers
    :param df: the df of the train_labels.csv
    :param mode:
    :return:
    """
    data_list = []
    for _, row in df.iterrows():
        subject_id = row["BraTS21ID"]
        label = row["MGMT_value"]
        data_list.append({
            T1: os.path.join(INPUT_DIR, mode, subject_id, T1),
            T2: os.path.join(INPUT_DIR, mode, subject_id, T2),
            T1GD: os.path.join(INPUT_DIR, mode, subject_id, T1GD),
            FLAIR: os.path.join(INPUT_DIR, mode, subject_id, FLAIR),
            LABEL: label
        })
    return data_list


class DicomSeries3DReader(ImageReader):
    def __init__(self) -> None:
        super().__init__()
        self.header = {}

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        return os.path.exists(filename) and os.path.isdir(filename)

    def read(self, data: Union[Sequence[str], str], **kwargs) -> Union[Sequence[Any], Any]:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(data)
        reader.SetFileNames(dicom_names)
        itk_image = reader.Execute()
        return itk_image

    def _prep_meta_dict(self, img: sitk.Image):
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
        self._prep_meta_dict(img)
        self._prep_affine()
        self._prep_spatial_shape(img)
        self.header["original_channel_dim"] = "no_channel"

    def get_data(self, img: sitk.Image) -> Tuple[np.ndarray, Dict]:
        self._prep_header(img)
        img_array = sitk.GetArrayViewFromImage(img)
        return img_array, self.header
