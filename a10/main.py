import sys
import rawpy
import glob
import numpy as np
burst_folder = sys.argv[1]

burst_img_paths = sorted(glob.glob(burst_folder + '/payload*.dng'))

burst_raw_images = []
bayer_patten = None
for burst_img_path in burst_img_paths:
    with rawpy.imread(burst_img_path) as raw:
        raw_image = raw.raw_image.copy()
        black_level = raw.black_level_per_channel
        raw_image_2x2 = raw_image.reshape(raw_image.shape[0] // 2, 2, raw_image.shape[1] // 2, 2)
        black_level = np.array(black_level).reshape(2,2)[None, :, None, :]
        raw_image_2x2 = np.maximum(raw_image_2x2 - black_level, 0)
        raw_image = raw_image_2x2.reshape(raw_image.shape[0], raw_image.shape[1])
        burst_raw_images.append(raw_image.copy())
        if bayer_patten is None:
            bayer_patten = "".join([chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten()])
