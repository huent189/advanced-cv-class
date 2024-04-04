import numpy as np
from helper import compute_gaussian_pyramid
import cv2

class TileBasedAlignment():
    def __init__(self) -> None:
        self.tileSizes = [16, 16, 16, 8]
        self.searchRadius = [4,4,4,4]
    
    def compute_motion_vector(self, ref_im, im, search_radius, tile_size, diff_mode):
        for shift_x in range(1, search_radius):
            for shift_y in range(1, search_radius):
                

    def align(self):
        # We implement this strategy by averaging
        # 2×2 blocks of Bayer RGGB samples, so that we align downsampled
        # 3 Mpix grayscale images instead of 12 Mpix raw images.
        downsampled_imgs = [img.reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2).mean(axis=(1, 3)) for img in self.imgs]
        
        # we perform a coarse-to-fine alignment on four-level
        # Gaussian pyramids of the downsampled-to-gray raw input
        ref_pyramid = compute_gaussian_pyramid(downsampled_imgs[self.ref_idx], self.num_levels)
        for i in range(len(self.imgs)):
            if i == self.ref_idx:
                continue
            pyr = compute_gaussian_pyramid(downsampled_imgs[i], self.num_levels)
            for j in range(self.num_levels):
                