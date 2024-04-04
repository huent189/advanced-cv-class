import numpy as np
import cv2
def compute_gaussian_pyramid(image, num_levels):
    gaussian_pyramid = [image]
    for _ in range(1, num_levels):
        downsampled_image = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(downsampled_image)
    return gaussian_pyramid
def compute_laplacian_pyramid(image, num_levels):
    gaussian_pyramid = compute_gaussian_pyramid(image, num_levels)
    laplacian_pyramid = []
    
    for i in range(1, num_levels, 1):
        # upsample by 2 and convolve with a 5x5 Gaussian filter
        dstsize = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        upsized_image = cv2.pyrUp(gaussian_pyramid[i], dstsize=dstsize)
        diff = cv2.subtract(gaussian_pyramid[i - 1], upsized_image)
        laplacian_pyramid.append(diff)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

