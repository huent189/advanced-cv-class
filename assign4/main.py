import rawpy
import numpy as np
import matplotlib.pyplot as plt
import sys

def padding(image, new_h, new_w):
    h, w = image.shape[:2]
    diff_h = new_h - h
    diff_w = new_w - w
    if len(image.shape) == 2:
        padded_image = np.zeros((new_h, new_w))
    else:
        padded_image = np.zeros((new_h, new_w, image.shape[2]))
    padded_image[diff_h // 2 :h + diff_h // 2, diff_w // 2 :diff_w // 2 + w] = image
    return padded_image

def demosaic_bilinear_interpolation(rgb_image_bayer_only, pattern_np):
    sum_over_window = 0
    weight = 0
    h, w = rgb_image_bayer_only.shape[:2]
    expanded_pattern = pattern_np[None, None].repeat(h // 2, axis=0).repeat(w // 2, axis=1)
    expanded_pattern = expanded_pattern.reshape(h // 2, w // 2, 2, 2, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, 3)
    for i in range(3):
        for j in range(3):
            sum_over_window += rgb_image_bayer_only[i:i - 2 + h, j:j-2 + w]
            weight += expanded_pattern[i:i - 2 + h, j:j-2 + w]
    interpolated_img = sum_over_window / weight
    h -= 2
    w -= 2
    rgb_image_bayer_only = rgb_image_bayer_only[1:-1, 1:-1]
    expanded_pattern = expanded_pattern[1:-1, 1:-1]
    rgb_image_demosaiced = rgb_image_bayer_only * expanded_pattern + interpolated_img * (1 - expanded_pattern)
    return rgb_image_demosaiced

def scale_and_gamma(image, gamma=2.2):
    scaled_image = image / np.max(image)
    gamma_corrected_image = scaled_image ** (1/gamma)
    return gamma_corrected_image

def plot_images(bayer_image, rgb_image, demosaiced_image):
    bayer_image = bayer_image ** (1 / 2.2)
    rgb_image = rgb_image ** (1 / 2.2)
    demosaiced_image = demosaiced_image ** (1 / 2.2)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(bayer_image, cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('Bayer Grey Pattern')
    axs[1].imshow(rgb_image, vmin=0, vmax=1)
    axs[1].set_title('Bayer RGB Pattern')
    axs[2].imshow(demosaiced_image, vmin=0, vmax=1)
    axs[2].set_title('Demosaiced Pattern')
    plt.show()

# Read the raw image and extract Bayer pattern
raw_image_path = sys.argv[1]
with rawpy.imread(raw_image_path) as raw:
    bayer_image = raw.raw_image.astype('uint16')[:-1, :-1]
    bayer_pattern = "".join([chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten()])
    print("Bayer pattern:", bayer_pattern)

bayer_image = bayer_image / np.max(bayer_image)
# print(bayer_image.max(), bayer_image.min())
# convert bayer_pattern to channel index array
channel_name_to_idx = {"R": [1, 0, 0], "G": [0, 1, 0], "B": [0, 0, 1]}
bayer_pattern_np = [channel_name_to_idx[c] for c in bayer_pattern]
bayer_pattern_np = np.array(bayer_pattern_np, dtype=np.uint16)

h, w = bayer_image.shape
pad_h, pad_w = h % 2, w % 2
bayer_image = padding(bayer_image, h + pad_h, w + pad_w)

h, w = bayer_image.shape
assert (h % 2 == 0) and (w % 2 == 0), "Image dimensions must be even"
#  RGB image where the missing color channels in the Bayer image are set to 0.
four_channels_bayer_image = bayer_image.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).reshape(h // 2, w // 2, 4)
four_channels_bayer_image = four_channels_bayer_image[..., None] * bayer_pattern_np[None, None, None,:]
rgb_image_bayer_only = four_channels_bayer_image.reshape(h // 2, w // 2, 2, 2, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, 3)

rgb_image_demosaiced = demosaic_bilinear_interpolation(rgb_image_bayer_only, bayer_pattern_np)

# print(bayer_pattern)
# print(rgb_image_bayer_only[:5, 1])
# print('------------------')
# print(rgb_image_bayer_only[1,:5])
# print('------------------')
# print(rgb_image_demosaiced[:4, 0])
# print('------------------')
# print(rgb_image_demosaiced[0,:4])
# print(rgb_image_bayer_only[1:4, 0:3])
# # Demosaic the Bayer image
# rgb_image_demosaiced = demosaic(bayer_image, bayer_pattern)
# rgb_image_demosaiced = scale_and_gamma(rgb_image_demosaiced)

# Plot the images
bayer_image = bayer_image[:h - pad_h, :w - pad_w]
rgb_image_bayer_only = rgb_image_bayer_only[:h - pad_h, :w - pad_w]
rgb_image_demosaiced = rgb_image_demosaiced[:rgb_image_demosaiced.shape[0] - pad_h, :rgb_image_demosaiced.shape[1] - pad_w]
rgb_image_demosaiced = padding(rgb_image_demosaiced, h - pad_h, w - pad_w)
plot_images(bayer_image, rgb_image_bayer_only, rgb_image_demosaiced)
