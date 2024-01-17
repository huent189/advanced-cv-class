import rawpy
import numpy as np
import cv2
def open_raw_img(path, normalize=True):
    with rawpy.imread(path) as raw:
        raw_image = raw.raw_image.copy()
        if normalize:
            raw_image = (raw_image - np.min(raw_image)) / (np.max(raw_image) - np.min(raw_image))
    return raw_image
def adjust_exposure(img, exposure):
    return img * exposure
def resize_raw_img(img, scale_factor):
    if len(img.shape) == 2:
        # bayer image
        new_h = int(img.shape[0] * scale_factor) // 2
        new_w = int(img.shape[1] * scale_factor) // 2
        c1 = img[0::2, 0::2]
        c2 = img[0::2, 1::2]
        c3 = img[1::2, 0::2]
        c4 = img[1::2, 1::2]
        resized_image = np.zeros((new_h * 2, new_w * 2), dtype=img.dtype)
        resized_image[0::2, 0::2] = cv2.resize(c1, (new_w, new_h))
        resized_image[0::2, 1::2] = cv2.resize(c2, (new_w, new_h))
        resized_image[1::2, 0::2] = cv2.resize(c3, (new_w, new_h))
        resized_image[1::2, 1::2] = cv2.resize(c4, (new_w, new_h))
    elif len(img.shape) == 3:
        # Foveon and other RGB-type image
        new_h = int(img.shape[0] * scale_factor)
        new_w = int(img.shape[1] * scale_factor)
        resized_image = cv2.resize(img, (new_w, new_h))
    else:
        raise ValueError('Image must be either 2D or 3D')
    return resized_image