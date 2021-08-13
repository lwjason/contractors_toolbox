import numpy as np
import matplotlib.pyplot as plt


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
