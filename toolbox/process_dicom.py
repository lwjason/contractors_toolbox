import glob
import os
import re

import SimpleITK as sitk
import pandas as pd
import pydicom
from natsort import natsorted

""" DICOM RELATED NAMED CONSTANTS USED WITHIN PROJECT
"""

DICOM_PATH = "path"
SUBJECT_ID = "subject_id"
SEQUENCE_NAME = "sequence_name"
TAGS_OF_INTEREST = [
        'Series Description',
        'MR Acquisition Type',
        'Slice Thickness',
        'Imaged Nucleus',
        'Number of Averages',
        'Echo Number(s)',
        'Magnetic Field Strength',
        'Spacing Between Slices',
        'In-plane Phase Encoding Direction',
        'Patient Position',
        'Photometric Interpretation',
        'Rows',
        'Columns',
        'Pixel Spacing',
        'Pixel Representation',  # Pixel Representation (0028,0103) is either unsigned (0) or signed (1)
        'Window Center',
        'Window Width',
        'Rescale Intercept',
        'Rescale Slope',
        'Rescale Type',
    ]


class DicomTagProcessor:
    """ Class for processing DICOM files and tags which are specific to
     RSNA-MICCAI Brain Tumor Radiogenomic Classification competition.
    """
    @staticmethod
    def _sanitise_unicode(s):
        return s.replace(u"\u0000", "").strip()

    @staticmethod
    def _map_multi_value(list_):
        new_list = list_
        if all(isinstance(i, str) for i in list_):
            if all(i.isdecimal() for i in list_):
                new_list = [float(i) for i in list_]
                if all(f.is_integer() for f in new_list):
                    new_list = [int(f) for f in new_list]
        return new_list

    @staticmethod
    def _get_dicom_df(dataset_path):
        """ Returns pandas.DataFrame object for specific DICOM file
        :param dataset_path: example: "/kaggle/input/*/train/*/*"
        :return: df of columns: subject_id, sequence_name, path
        """
        data_list = []
        for sequence_path in glob.glob(dataset_path):
            *_, subject_id, sequence_name = sequence_path.split("/")
            first_dcm_file = get_path_to_first_dicom_file(sequence_path)
            data_list.append({
                SUBJECT_ID: subject_id,
                SEQUENCE_NAME: sequence_name,
                DICOM_PATH: first_dcm_file
            })
        df = pd.DataFrame(data_list)
        return df

    def _convert_dicom_tag_type(self, element):
        mapping = {
            list: lambda x: x,
            int: lambda x: x,
            float: lambda x: x,
            str: self._sanitise_unicode,
            bytes: lambda x: self._sanitise_unicode(x.decode('ascii', 'replace')),
            pydicom.uid.UID: str,
            pydicom.valuerep.PersonName: str,
            pydicom.valuerep.DSfloat: float,
            pydicom.valuerep.IS: int,
            pydicom.multival.MultiValue: self._map_multi_value,
        }
        value = element.value
        if value is None:
            return
        for type_, func in mapping.items():
            if type(value) is type_:
                return func(value)

    def get_elements(self, ds):
        content = {}
        for de in ds.iterall():
            value = self._convert_dicom_tag_type(de)
            if de.name not in content:
                content[de.name] = value
            else:
                raise Exception(f"There is a duplicate dicom tag, please check the logic: \n {de.name}: {value}")
        return content

    def get_tags_of_interest(self, dicom_file, tags_of_interest):
        keys, subject_list = tags_of_interest + [SUBJECT_ID, SEQUENCE_NAME], []
        for _, row in DicomTagProcessor._get_dicom_df(dicom_file).iterrows():
            ds = pydicom.dcmread(row[DICOM_PATH], stop_before_pixels=True)
            elements = self.get_elements(ds)
            data = {
                SUBJECT_ID: row[SUBJECT_ID],
                SEQUENCE_NAME: row[SEQUENCE_NAME],
            }
            data.update(elements)
            subject_list.append(data)
        return pd.DataFrame([{k: v for k, v in s.items() if k in keys} for s in subject_list])


def get_file_number(path):
    """ Return DICOM file number.
    :param path: absolute path to DICOM file.
    :return: slice number - integer
    """
    return int(re.findall(r'\d+', path)[0])


def get_path_to_first_dicom_file(dicom_folder):
    """ Return absolute path to first DICOM file (i.e. first slice)
    :param dicom_folder: absolute path to directory which contains list of
    subject specific DICOM files.
    :return: absolute DICOM file path that corresponds to the first slice.
    """
    min_number_file = [99999999, None]
    for dcm_file in glob.glob(os.path.join(dicom_folder, "*")):
        file_number = get_file_number(dcm_file)
        if file_number < min_number_file[0]:
            min_number_file[0] = file_number
            min_number_file[1] = dcm_file
    return min_number_file[1]


def get_path_to_middle_file(dicom_folder):
    """ Return absolute path to middle DICOM file (i.e. middle slice)
    :param dicom_folder: absolute path to directory which contains list of
    subject specific DICOM files.
    :return: absolute DICOM file path that corresponds to the middle slice.
    """
    slice_ids_dict = {}
    for dcm in os.listdir(dicom_folder):
        slice_id = get_file_number(dcm)
        slice_ids_dict[slice_id] = os.path.join(dicom_folder, dcm)
    slice_ids = natsorted(list(slice_ids_dict.keys()))
    return slice_ids_dict[slice_ids[len(slice_ids)//2]]


def load_scan(dicom_dir_path):
    """ Read DICOM images from specified directory and
    sort them by Slice Location

    :param dicom_dir_path - absolute path to dicom directory.
    :return list of slices sorted by Slice Location
    """
    if not os.path.exists(dicom_dir_path):
        raise NotADirectoryError(f"{dicom_dir_path} is not a directory!")
    slices = [pydicom.dcmread(os.path.join(dicom_dir_path, s)) for s in os.listdir(dicom_dir_path)]
    slices.sort(key=lambda s: s.SliceLocation)
    return slices


class DicomReader:
    """Read a dicom series by SimpleITK"""
    def __init__(self, dcm_root: str):
        """ Init the class by a given dcm_root.

        :param dcm_root: a folder path, it should be the parent folder of a dicom series,
        for example /kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00000/FLAIR/.
        """
        self.dcm_root = dcm_root
        self.itk_image = read_from_dicom_series(dcm_root)

    @staticmethod
    def read_from_dicom_series(dcm_root) -> sitk.ImageSeriesReader:
        """Read a dicom series by SimpleITK and return sitk_image

        :param dcm_root: image root of each seq root
        :return: sitk image
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dcm_root)
        reader.SetFileNames(dicom_names)
        itk_image = reader.Execute()
        return itk_image

    def get_itk_image(self):
        """ Returns SimpleITK image object.
        """
        return self.itk_image

    def get_array(self):
        """ Returns NumPy N-dimensional array from SimpleITK image.
        """
        return sitk.GetArrayFromImage(self.itk_image)

    def print_image_info(self):
        """ Prints SimpleITK image information"""
        print(f"[INFO]: Shape - {self.itk_image.GetSize()}")
        print(f"[INFO]: Spacing - {self.itk_image.GetSpacing()}")
        print(f"[INFO]: Origin - {self.itk_image.GetOrigin()}")
        print(f"[INFO]: Direction - {self.itk_image.GetDirection()}\n")
