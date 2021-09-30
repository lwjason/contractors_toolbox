import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.notebook import tqdm_notebook
from toolbox.constants import T1, T1GD, T2, FLAIR
import random


class VisualizeMedicalDataset:
    def __init__(self, data: tio.SubjectsDataset, transforms: tio.Compose, dataset_type: str):
        self.data = data
        self.transforms = transforms
        self.dataset_type = dataset_type
        # self.max_value = self.get_max_value()
        # self.min_value = self.get_min_value()

    def plot_histogram(self, axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
        values = tensor.numpy().ravel()
        kernel = stats.gaussian_kde(values)
        positions = np.linspace(values.min(), values.max(), num=num_positions)
        histogram = kernel(positions)
        kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
        if label is not None:
            kwargs['label'] = label
        axis.plot(positions, histogram, **kwargs)

    def compute_histograms(self):
        fix, ax = plt.subplots(dpi=100)
        for subject in tqdm_notebook(self.data.dry_iter()):
            tensor = subject.get_first_image().data
            if T1 in subject.__dict__.keys():
                color = 'red'
            elif T1GD in subject.__dict__.keys():
                color = 'yellow'
            elif T2 in subject.__dict__.keys():
                color = 'green'
            elif FLAIR in subject.__dict__.keys():
                color = 'blue'
            self.plot_histogram(ax, tensor, color=color)
        # ax.set_xlim(self.min_value, self.max_value)
        ax.set_xlim(-100, 5000)
        ax.set_ylim(0, 0.004)
        ax.set_title(f"Intensity Histogram for {self.dataset_type} Dataset")
        ax.set_xlabel("Intensity Value")
        ax.grid()

    def plot_subject(self, subject_id=None, output_path=None):
        """ Plots subject that corresponds to the specified subject ID, or
        random if not specfiied..

        :param subject_id - string by which subjects are uniquely identified.
        """
        if subject_id:
            subject_idx = [idx for idx, subject in enumerate(self.data.dry_iter(), 0)
                           if subject.__dict__['subject_id'] == subject_id][0]
            if not subject_idx:
                raise ValueError(f"[ERROR]: Subject with {subject_id} does not exist in the dataset!")
        else:
            subject_ids = [subject.__dict__['subject_id'] for subject in self.data.dry_iter()]
            subject_idx = random.randint(0, len(subject_ids))
        self.data[subject_idx].plot(output_path=output_path)

    def get_max_value(self):
        """ Returns maximum pixel value based on list of subjects

        :param subjects - list of subjects for which we want
         to retrieve maximum value
        :retunr maximum subject value
        """
        values = np.array([image.numpy().max() for s in self.data for image in s.get_images()])
        return values.max(axis=0)

    def get_min_value(self):
        """ Returns maximum pixel value based on list of subjects

        :param subjects - list of subjects for which we want
         to retrieve maximum value
        :retunr maximum subject value
        """
        values = np.array([image.numpy().min() for s in self.data for image in s.get_images()])
        return values.min(axis=0)