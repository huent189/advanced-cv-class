import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.widgets import Slider
import cv2
import skimage
BLOCK_SIZE = 8
ZIGZAG_PATTERN = np.array([[0,  1,  5,  6,  14, 15, 27, 28],
               [2,  4,  7,  13, 16, 26, 29, 42],
               [3,  8,  12, 17, 25, 30, 41, 43],
               [9,  11, 18, 24, 31, 40, 44,53],
               [10, 19, 23, 32, 39, 45, 52,54],
               [20, 22, 33, 38, 46, 51, 55,60],
               [21, 34, 37, 47, 50, 56, 59,61],
               [35, 36, 48, 49, 57, 58, 62,63]])
CACHED_VALUES = None    
dct_block_idx = (31, 31)

def dct2(image, block_size=8):
    assert len(image.shape) == 2
    h, w = image.shape
    image = image.reshape(h // block_size, block_size, w // block_size, block_size)
    image = image.transpose(0, 2, 1, 3)
    return scipy.fft.dctn(image, axes=(-1,-2), norm='ortho')
def idct2(image, block_size=8):
    h, w = image.shape[:2]
    image = scipy.fft.idctn(image, axes=(-1,-2), norm='ortho')
    image = image.transpose(0, 2, 1, 3)
    return image.reshape(h * block_size, w * block_size)

def pad_image(image, block_size=8):
    h, w = image.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
def unpad_image(image, h, w):
    return image[:h, :w]
def compress(yuv, reduction_factor, im_h, im_w, uv_reduction_factor=4, use_cache=True):
    global CACHED_VALUES

    if use_cache and CACHED_VALUES is not None:
        y_dct, u_dct, v_dct = CACHED_VALUES
    else:
        y, u, v = yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]
        u = scipy.ndimage.zoom(u, (1.0 / uv_reduction_factor, 1.0 / uv_reduction_factor), mode='reflect', grid_mode=True)
        v = scipy.ndimage.zoom(v, (1.0 / uv_reduction_factor, 1.0 / uv_reduction_factor), mode='reflect', grid_mode=True)
        y = pad_image(y, BLOCK_SIZE)
        u = pad_image(u, BLOCK_SIZE)
        v = pad_image(v, BLOCK_SIZE)
        y_dct = dct2(y)
        u_dct = dct2(u)
        v_dct = dct2(v)
        CACHED_VALUES = (y_dct, u_dct, v_dct)
    # Quantize DCT coefficients
    
    mask = np.ones_like(ZIGZAG_PATTERN)
    threshold = (BLOCK_SIZE ** 2) - reduction_factor
    mask[ZIGZAG_PATTERN >= threshold] = 0
    y_dct_quant = y_dct * mask
    u_dct_quant = u_dct * mask
    v_dct_quant = v_dct * mask

    y_rec = idct2(y_dct_quant)
    u_rec = idct2(u_dct_quant)
    v_rec = idct2(v_dct_quant)
    y_rec = unpad_image(y_rec, im_h, im_w)
    uv_h, uv_w = int(round(im_h / uv_reduction_factor)), int(round(im_w / uv_reduction_factor))
    u_rec = unpad_image(u_rec, uv_h, uv_w)
    v_rec = unpad_image(v_rec, uv_h, uv_w)
    u_rec = scipy.ndimage.zoom(u_rec, (uv_reduction_factor, uv_reduction_factor), mode='reflect', grid_mode=True)
    v_rec = scipy.ndimage.zoom(v_rec, (uv_reduction_factor, uv_reduction_factor), mode='reflect', grid_mode=True)

    yuv_rec = np.dstack((y_rec, u_rec, v_rec))
    rgb_output = skimage.color.yuv2rgb(yuv_rec)
    return rgb_output.clip(0,1),mask, y_dct_quant, (y_rec, u_rec, v_rec)

def rmse(original, reconstructed):
    diff = original - reconstructed
    per_pixel_rmse = np.sqrt(np.mean(diff**2, axis=-1))
    rmse = np.sqrt(np.mean(diff**2))
    return per_pixel_rmse, rmse

if __name__ == '__main__':
    import sys
    image_path = sys.argv[1]
    uv_reduction_factor = 4
    img = cv2.imread(image_path)  # Read image using ndimage for flexibility
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    yuv = skimage.color.rgb2yuv(rgb_img)

    rgb_output, dct_mask, y_dct_quant, yuv_compressed = compress(yuv, 0, yuv.shape[0], yuv.shape[1], uv_reduction_factor=uv_reduction_factor)
    

    rmse_image, rmse_val = rmse(rgb_img, rgb_output)
    fig, axes = plt.subplots(2, 4)
    compressed_fig = axes[0, 0].imshow(rgb_output, aspect='equal')
    axes[0, 0].set_title('Compressed')
    axes[0, 0].set_axis_off()

    y_fig = axes[0, 1].imshow(yuv_compressed[0], cmap='gray', aspect='equal')
    axes[0, 1].set_title('Luma')
    axes[0, 1].set_axis_off()

    u_fig = axes[0, 2].imshow(yuv_compressed[1], cmap='gray', aspect='equal')
    axes[0, 2].set_title('1/4 U')
    axes[0, 2].set_axis_off()

    v_fig = axes[0, 3].imshow(yuv_compressed[2], cmap='gray', aspect='equal')
    axes[0, 3].set_title('1/4 V')
    axes[0, 3].set_axis_off()

    axes[1, 0].imshow(rgb_img, aspect='equal')
    axes[1, 0].set_title('Original')
    axes[1, 0].set_axis_off()

    rmse_fig = axes[1, 1].imshow(rmse_image, aspect='equal', vmin=0, vmax=1.0)
    axes[1, 1].set_title(f'RMSE {rmse_val:.4f}')
    axes[1, 1].set_axis_off()

    dct_mask_fig = axes[1, 2].imshow(dct_mask, cmap='gray', vmin=0, vmax=1.0, aspect='equal')
    axes[1, 2].set_title(f'DCT coefficients used: {BLOCK_SIZE * BLOCK_SIZE}')
    # Set the ticks and grid
    axes[1, 2].set_xticks(np.arange(-0.5, 8, 1), minor=True)
    axes[1, 2].set_yticks(np.arange(-0.5, 8, 1), minor=True)
    axes[1, 2].grid(which="minor", color="gray", linestyle='-', linewidth=2)
    axes[1, 2].tick_params(which="minor", size=0)
    axes[1, 2].set_xticklabels([])
    axes[1, 2].set_yticklabels([])
    axes[1, 2].tick_params(axis='both', which='both', length=0)

    dct_block = y_dct_quant[dct_block_idx[0], dct_block_idx[1]]
    dct_block_fig = axes[1, 3].imshow(np.log(np.abs(dct_block) + 1e-6))
    axes[1, 3].set_title(f'DCT Coefficients of Block [{dct_block_idx[0]}, {dct_block_idx[1]}]')
    axes[1, 3].set_axis_off()
    # Slider for reduction factor
    ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03])
    slider = Slider(ax=ax_slider, label='Reduction Factor', valmin=0, valmax=63, valinit=0, valstep=1)
    def update(val):
        global y_dct_quant
        reduction_factor = int(val)
        rgb_output, dct_mask, y_dct_quant, yuv_compressed = compress(yuv, reduction_factor, yuv.shape[0], yuv.shape[1], uv_reduction_factor=uv_reduction_factor, use_cache=True)
        rmse_image, rmse_val = rmse(rgb_img, rgb_output)
        
        dct_block = y_dct_quant[dct_block_idx[0], dct_block_idx[1]]
        compressed_fig.set_data(rgb_output)
        rmse_fig.set_data(rmse_image)
        dct_block_fig.set_data(np.log(np.abs(dct_block) + 1e-6))
        dct_mask_fig.set_data(dct_mask)
        y_fig.set_data(yuv_compressed[0])
        u_fig.set_data(yuv_compressed[1])
        v_fig.set_data(yuv_compressed[2])
        axes[1, 1].set_title(f'RMSE {rmse_val:.4f}')
        axes[1, 2].set_title(f'DCT coefficients used: {BLOCK_SIZE * BLOCK_SIZE - reduction_factor}')
        plt.draw()
    slider.on_changed(update)

    def on_click(event):
        global dct_block_idx
        if event.inaxes is axes[0, 0]:
            x, y = event.xdata, event.ydata
            block_x = int(np.floor(y / BLOCK_SIZE))
            block_y = int(np.floor(x / BLOCK_SIZE))
            dct_block_idx = (block_x, block_y)
            dct_block = y_dct_quant[dct_block_idx[0], dct_block_idx[1]]
            dct_block_fig.set_data(np.log(np.abs(dct_block) + 1e-6))
            axes[1, 3].set_title(f'DCT Coefficients of Block [{dct_block_idx[0]}, {dct_block_idx[1]}]')
            plt.draw()
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()