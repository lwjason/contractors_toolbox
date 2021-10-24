import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from .load_image import read_rgb_image
from .mask import get_masks


def print_img_with_mask(input_img, mask, title=None):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img2 = clahe.apply(input_img)
    img3 = cv2.equalizeHist(input_img)
    img = np.stack([input_img, img2, img3], axis=-1)
    print(f"Labels: {np.unique(mask)}")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))

    axes[0].imshow(img)
    axes[0].set_title('Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='inferno')
    axes[1].set_title('Mask', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(img)
    axes[2].imshow(mask, alpha=0.4, cmap='inferno')
    axes[2].set_title('Image + Mask', fontsize=12)
    axes[2].axis('off')
    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=15, y=0.7)
    plt.show()


def print_subject_image(df, subject_id, root_dir, train_or_test):
    subject_1_df = df[df.id == subject_id]
    mask = get_masks(subject_1_df)
    img_filename = os.path.join(root_dir, f'{train_or_test}/{subject_id}.png')
    img = read_rgb_image(img_filename)[..., 0]
    print_img_with_mask(img, mask, title=f"id: {subject_id}")