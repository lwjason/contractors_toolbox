import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchio as tio

from PyPDF2 import PdfFileMerger
from matplotlib.backends.backend_pdf import PdfPages
from toolbox.logger import logging
from toolbox.process_dicom import DicomReader

train_image_root = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train'
test_image_root = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test'


class VisualizeMedicalDataset():
    def __init__(self, data: tio.SubjectsDataset,  dataset_type: str):
        """ VisaulizeMedicalDataset constructor

        :param data: torchio.SubjectsDataset which we want to visualize.
        :param dataset_type: dataset type (i.e., training, validation)
        """
        self.data = data
        self.dataset_type = dataset_type
        # self.max_value = self.get_max_value()
        # self.min_value = self.get_min_value()

    @staticmethod
    def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
        """ Plots Histogram.

        :param axis: matplotlib axis
        :param num_positions: number of positions
        :param label: plot label
        :param alpha: opacity
        :param color: color of the plot
        """
        values = tensor.numpy().ravel()
        kernel = stats.gaussian_kde(values)
        positions = np.linspace(values.min(), values.max(), num=num_positions)
        histogram = kernel(positions)
        kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
        if label is not None:
            kwargs['label'] = label
        axis.plot(positions, histogram, **kwargs)

    def compute_histograms(self):
        """ Performs Histogram Standardization of intensity values.
        """
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
            plot_histogram(ax, tensor, color=color)
        # ax.set_xlim(self.min_value, self.max_value)
        ax.set_xlim(-100, 5000)
        ax.set_ylim(0, 0.004)
        ax.set_title(f"Intensity Histogram for {self.dataset_type} Dataset")
        ax.set_xlabel("Intensity Value")
        ax.grid()

    def plot_subject(self, subject_id=None):
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
        self.data[subject_idx].plot()

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


def show_n_slices(array, start_from=0, step=10, columns=6, figsize=(18,10)):
    """ Plot N slices of a 3D image.
    :param array: N-dimensional numpy array (i.e. 3D image)
    :param start_from: slice index to start from
    :param step: step to take when moving to the next slice
    :param columns: number of columns in the plot
    :param figsize: figure size in inches
    """
    array = np.swapaxes(array, 0, 2)
    fig, ax = plt.subplots(1, columns, figsize=figsize)
    slice_num = start_from
    for n in range(columns):
        ax[n].imshow(array[:, :, slice_num], 'gray')
        ax[n].set_xticks([])
        ax[n].set_yticks([])
        ax[n].set_title('Slice number: {}'.format(slice_num), color='r')
        slice_num += step

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def draw_or_save(draw, save, pdf):
    if draw:
        plt.show()
    if save:
        pdf.savefig()
        plt.close()


def save_image_to_pdf(subject, seq, image_root, output_folder, save=True, overwrite=True):
    """
    Save pdf by given path

    :param subject:
    :param seq:
    :param image_root:
    :param output_folder:
    :param save:
    :param overwrite:
    :return:
    """
    # https://stackoverflow.com/questions/18266642/multiple-imshow-subplots-each-with-colorbar
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{subject}_{seq}.pdf"
    pdf_filename = os.path.join(output_folder, filename)
    if not overwrite:
        if os.path.exists(pdf_filename):
            logging.info(f"{pdf_filename} exists, skip!")
            return
    dcm_image_root = os.path.join(image_root, subject, seq)
    dcm_image = DicomReader(dcm_image_root)
    dcm_array = dcm_image.get_array()
    slice_num = dcm_array.shape[0]
    display_dcm_array = [(i, dcm_array[i, :, :]) for i in range(int(3*slice_num/10), int(8*slice_num/10), int(slice_num/10))]
    fig_x, fig_y = (1, 5)
    fig, axes = plt.subplots(fig_x, fig_y, figsize=(fig_y * 3, fig_x * 3.5))
    for idx, (image_slice, pixel_array) in enumerate(display_dcm_array):
        axes.flatten()[idx].set_title(f"slice: {image_slice}")
        axes.flatten()[idx].imshow(pixel_array, cmap="gray")
        axes.flatten()[idx].xaxis.set_visible(False)
        axes.flatten()[idx].yaxis.set_visible(False)

    fig.suptitle(
        f"subject: {subject}, seq: {seq}, image_shape: "
        f"{dcm_array.shape}", fontsize=12, y=0.98)
    try:
        if overwrite:
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)
        with PdfPages(pdf_filename) as pdf:
            logging.info(f"Saving {pdf_filename}")
            draw_or_save(draw=False, save=save, pdf=pdf)
    except Exception as e:
        logging.error(e)


def pdf_cat(pdfs, output_filename):
    """

    :param pdfs: List of pdf, for example [pdf_path_1, pdf_path_2]
    :param output_filename:
    :return:
    """
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(output_filename)
    merger.close()
