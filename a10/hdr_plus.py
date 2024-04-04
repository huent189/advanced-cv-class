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
        self.tileSizes = [16, 16, 16, 8]
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
    def pad(self, images, tile_size):
        h, w = images[0].shape[:2]
        # if needed, pad images with zeros so that getTiles contains all image pixels
        paddingPatchesHeight = (tile_size - h % (tile_size)) * (h % (tile_size) != 0)
        paddingPatchesWidth = (tile_size - w % (tile_size)) * (w % (tile_size) != 0)
        # additional zero padding to prevent artifacts on image edges due to overlapped patches in each spatial dimension
        paddingOverlapHeight = paddingOverlapWidth = tile_size // 2
        # combine the two to get the total padding
        paddingTop = paddingOverlapHeight
        paddingBottom = paddingOverlapHeight + paddingPatchesHeight
        paddingLeft = paddingOverlapWidth
        paddingRight = paddingOverlapWidth + paddingPatchesWidth
        # pad all images (by mirroring image edges)
        imagesPadded = [np.pad(im, ((paddingTop, paddingBottom), (paddingLeft, paddingRight)), 'symmetric') for im in images]
        return imagesPadded
    def unpad(self, image, tile_size, h, w):
        paddingOverlapHeight = paddingOverlapWidth = tile_size // 2
        return image[paddingOverlapHeight:paddingOverlapHeight+h, paddingOverlapWidth:paddingOverlapWidth+w]
    