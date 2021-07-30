import re
import glob
import os
import pandas as pd
import pydicom

SUBJECT_ID = "subject_id"
SEQUENCE_NAME = "sequence_name"


class Sequence:
    FLAIR = "FLAIR"
    T1 = "T1w"
    T1GD = "T1wCE"
    T2 = "T2"
    ALL = [FLAIR, T1, T1GD, T2]


def get_file_number(path):
    return int(re.findall(r'\d+', path)[0])


def get_first_dicom_file(dicom_folder):
    min_number_file = [99999999, None]
    for dcm_file in glob.glob(os.path.join(dicom_folder, "*")):
        file_number = get_file_number(dcm_file)
        if file_number < min_number_file[0]:
            min_number_file[0] = file_number
            min_number_file[1] = dcm_file
    return min_number_file[1]


def get_first_file_df(dataset_path):
    """
    :param dataset_path: example: "/kaggle/input/*/train/*/*"
    :return: df of columns:
        subject_id, sequence_name, path
    """
    data_list = []
    for sequence_path in glob.glob(dataset_path):
        *_, subject_id, sequence_name = sequence_path.split("/")
        first_dcm_file = get_first_dicom_file(sequence_path)
        data_list.append({
            SUBJECT_ID: subject_id,
            SEQUENCE_NAME: sequence_name,
            "path": first_dcm_file
        })
    df = pd.DataFrame(data_list)
    return df


class DicomTagProcessor:
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
        else:
            print(element)
            raise Exception(type(value))

    def get_elements(self, ds):
        content = {}
        for de in ds.iterall():
            value = self._convert_dicom_tag_type(de)
            if de.name not in content:
                content[de.name] = value
            else:
                raise Exception(f"There is a duplicate dicom tag, please check the logic: \n {de.name}: {value}")
        return content


class InterestedDicomTagExtractor:
    tag_of_interested = [
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
    keys = tag_of_interested + [SUBJECT_ID, SEQUENCE_NAME]

    @classmethod
    def get_df(cls, df_dcm_file):
        dicom_tag_processor = DicomTagProcessor()
        subject_list = []
        for _, row in df_dcm_file.iterrows():
            ds = pydicom.dcmread(row["path"], stop_before_pixels=True)
            elements = dicom_tag_processor.get_elements(ds)
            data = {
                "subject_id": row["subject_id"],
                "sequence_name": row["sequence_name"],
            }
            data.update(elements)
            subject_list.append(data)
        return pd.DataFrame([{k: v for k, v in s.items() if k in cls.keys} for s in subject_list])
