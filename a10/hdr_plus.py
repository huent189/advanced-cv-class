import cv2
import numpy as np
from helper import compute_gaussian_pyramid
pattern_map = {
    'RGGB': {'R': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'B': (1, 1)},
    'BGGR': {'B': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'R': (1, 1)},
    'GRBG': {'G1': (0, 0), 'R': (0, 1), 'B': (1, 0), 'G2': (1, 1)},
    'GBRG': {'G1': (0, 0), 'B': (0, 1), 'R': (1, 0), 'G2': (1, 1)},
}
class HDRPlus():
    def __init__(self) -> None:
        self.num_levels = 4
    def __call__(self, imgs, bayer_pattern):
        self.imgs = imgs
        self.bayer_pattern =pattern_map[bayer_pattern]
        self.select_reference()
    def compute_sharpness(self, raw_image):
        height, width = raw_image.shape
        green_channel = (raw_image[self.bayer_pattern['G1'][0]:height:2, self.bayer_pattern['G1'][1]:width:2] + 
                                 raw_image[self.bayer_pattern['G2'][0]:height:2, self.bayer_pattern['G2'][1]:width:2]) / 2
        sobel_x = cv2.Sobel(green_channel, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(green_channel, cv2.CV_32F, 0, 1, ksize=3)
        sobel_energy = np.sqrt(sobel_x**2 + sobel_y**2).mean()
        return sobel_energy
    def select_reference(self):
        candidates = self.imgs[:3]
        sharpness = [self.compute_sharpness(img) for img in candidates]
        self.ref_idx = np.argmax(sharpness)
    def sub_pixel_l2_align(self, ref_img, img):
        pass
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
                # compute the shift between the two images
                shift = cv2.phaseCorrelate(pyr[j], ref_pyramid[j])
                # apply the shift to the original raw image
                M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
                self.imgs[i] = cv2.warpAffine(self.imgs[i], M, (self.imgs[i].shape[1], self.imgs[i].shape[0]))