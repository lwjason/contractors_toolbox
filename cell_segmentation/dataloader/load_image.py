import cv2
import numpy as np

def read_img(filename, image_size=None):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    if img.max() > 255:
        img_max = img.max() # TODO: check
        img = (img/img_max).astype('uint8')
    return img


# image loader, using rgb only here
def read_rgb_image(filename, image_size=None):
    img = read_img(filename, image_size)
    stacked_images = np.stack([img for _ in range(3)],axis=-1)
    return stacked_images
