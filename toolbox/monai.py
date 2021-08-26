import os
from typing import Union, Sequence, Any, Tuple, Dict

import SimpleITK as sitk
import numpy as np
from monai.data import ImageReader


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
