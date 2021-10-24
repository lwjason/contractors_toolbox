import numpy as np

DEFAULT_IMAGE_SHAPE = (520, 704)
CELL_TYPE_LABEL = {
    "shsy5y": 1,
    "cort": 2,
    "astro": 3
}


def rle2mask(rle, label=1, shape=DEFAULT_IMAGE_SHAPE):
    '''Create mask by a given annotation.
    annotation: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


def get_masks(df):
    """Create a segmentation mask for a subject
    """
    if len(df.id.unique()) > 1:
        raise ValueError("There are multiple subjects in the input dataframe. Only accept 1 subject.")
    gt_masks = []
    for _, row in df.iterrows():
        rle = row.annotation
        label = CELL_TYPE_LABEL[row.cell_type]
        gt_mask = rle2mask(rle, label=label)
        gt_masks.append(gt_mask)
    individual_masks = np.stack(gt_masks)
    return np.max(individual_masks, axis=0)
