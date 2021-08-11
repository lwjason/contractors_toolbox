from toolbox.logger import logging
import os
from PyPDF2 import PdfFileMerger
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from toolbox.dicom import DicomReader

train_image_root = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train'
test_image_root = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test'


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
