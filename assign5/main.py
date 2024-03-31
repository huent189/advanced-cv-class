import rawpy
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.widgets import Button
from functools import partial
import matplotlib.patches as patches
img_plot = None
illumination_vector_plot = None
def read_dng(filename):
    # Open the DNG file and read the raw image data
    with rawpy.imread(filename) as raw:
        raw_image = raw.raw_image.copy()
        black_level = raw.black_level_per_channel
        bayer_pattern = "".join([chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten()])
        return raw_image, black_level, bayer_pattern
def simple_demosaic(raw_image, black_level, bayer_pattern='RGGB'):
    # Subtract the black level
    raw_image_2x2 = raw_image.reshape(raw_image.shape[0] // 2, 2, raw_image.shape[1] // 2, 2)
    black_level = np.array(black_level).reshape(2,2)[None, :, None, :]
    raw_image_2x2 = np.maximum(raw_image_2x2 - black_level, 0)
    raw_image = raw_image_2x2.reshape(raw_image.shape[0], raw_image.shape[1])

    # Simple demosaicing by down-sampling
    height, width = raw_image.shape
    demosaiced_image = np.zeros((height // 2, width // 2, 3), dtype=np.float32)
    # Map the Bayer pattern to indices
    pattern_map = {
        'RGGB': {'R': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'B': (1, 1)},
        'BGGR': {'B': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'R': (1, 1)},
        'GRBG': {'G1': (0, 0), 'R': (0, 1), 'B': (1, 0), 'G2': (1, 1)},
        'GBRG': {'G1': (0, 0), 'B': (0, 1), 'R': (1, 0), 'G2': (1, 1)},
    }

    pattern = pattern_map.get(bayer_pattern)
    if not pattern:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
    
    # Assign channels based on the Bayer pattern and black level subtraction
    demosaiced_image[:, :, 0] = raw_image[pattern['R'][0]:height:2, pattern['R'][1]:width:2]  # R channel
    demosaiced_image[:, :, 1] = (raw_image[pattern['G1'][0]:height:2, pattern['G1'][1]:width:2] + 
                                 raw_image[pattern['G2'][0]:height:2, pattern['G2'][1]:width:2]) / 2  # G channel
    demosaiced_image[:, :, 2] = raw_image[pattern['B'][0]:height:2, pattern['B'][1]:width:2]  # B channel
    demosaiced_image = demosaiced_image / np.max(demosaiced_image)
    return demosaiced_image

def compute_illumination_vector(demosaiced_image, trim_percent=5):
    # Flatten each channel for easier processing
    r_channel = demosaiced_image[:, :, 0].flatten()
    g_channel = demosaiced_image[:, :, 1].flatten()
    b_channel = demosaiced_image[:, :, 2].flatten()

    # Function to compute trimmed mean
    def trimmed_mean(channel, trim_percent):
        lower_bound = np.percentile(channel, trim_percent)
        upper_bound = np.percentile(channel, 100 - trim_percent)
        trimmed = channel[(channel >= lower_bound) & (channel <= upper_bound)]
        return np.mean(trimmed)

    # Compute the trimmed mean for R, G, and B channels
    avg_r = trimmed_mean(r_channel, trim_percent)
    avg_g = trimmed_mean(g_channel, trim_percent)
    avg_b = trimmed_mean(b_channel, trim_percent)

    # Compute the illumination vector using the gray world algorithm
    illumination_vector = np.array([avg_r, avg_g, avg_b])
    
    return illumination_vector

def apply_white_balance_correction(demosaiced_image, illumination_vector):
    # Diagonal white-balance correction
    illumination_vector = (illumination_vector + 1e-3)
    illumination_vector = illumination_vector[1] / illumination_vector  # Normalize the green channel
    corrected_image = demosaiced_image.copy()
    for i in range(3):  # Apply correction for each channel
        corrected_image[:, :, i] = demosaiced_image[:, :, i] * illumination_vector[i]
    corrected_image = np.clip(corrected_image, 0, 1)  # Ensure the values are within valid RGB range
    return corrected_image

def show_white_balance_correction(img_ax, illumination_vector_ax, demosaiced_image, illumination_vector):
    global img_plot
    global illumination_vector_plot
    color_block = np.ones((50, 50, 3))
    if illumination_vector is not None:
        for i in range(3):
            color_block[:, :, i] *= illumination_vector[i]
    color_block = np.clip(color_block, 0, 1)  # Ensure the values are within valid RGB range
    color_block = np.power(color_block, 1/2.2)
    # add black border to the color block
    color_block[0:2] = 0
    color_block[:,0:2] = 0
    color_block[-2:] = 0
    color_block[:,-2:] = 0
    # Add the color block to the right side of the image
    if illumination_vector_plot is None:
        illumination_vector_plot = illumination_vector_ax.imshow(color_block, vmin=0, vmax=1)
        illumination_vector_ax.title.set_text('Current \nillumination vector')
        illumination_vector_ax.axis('off')
        
    else:
        illumination_vector_plot.set_data(color_block)
    if illumination_vector is None:
        corrected_image = demosaiced_image
    else:
        corrected_image = apply_white_balance_correction(demosaiced_image, illumination_vector)
    display_image = np.power(corrected_image, 1/2.2)
    if img_plot is None:
        img_plot = img_ax.imshow(display_image, vmin=0, vmax=1)
        img_ax.axis('off')
    else:
        img_plot.set_data(display_image)
    plt.draw()

def interactive_white_balance_correction(event, img_ax, illuminatiove_vector_ax, demosaiced_image):
    if event.inaxes == img_ax:
        x, y = int(event.x), int(event.y)
        clicked_illumination = demosaiced_image[y, x]
        show_white_balance_correction(img_ax, illuminatiove_vector_ax, demosaiced_image, clicked_illumination)
def main(filename):
    # Step 1: Read the DNG file and extract the raw image
    raw_image, black_level, bayer_pattern = read_dng(filename)
    
    # Step 2: Demosaic the image
    demosaiced_image = simple_demosaic(raw_image, black_level, bayer_pattern)
    # Step 3: Perform auto white balance to compute the illumination vector
    awb_vector = compute_illumination_vector(demosaiced_image)

    fig = plt.figure(figsize=(11, 6))
    img_ax = plt.axes([0.05, 0.2, 0.7, 0.7])
    illuminatiove_vector_ax = plt.axes([0.8, 0.6, 0.1, 0.1])
    plt.subplots_adjust(bottom=0.2)
    show_white_balance_correction(img_ax, illuminatiove_vector_ax, demosaiced_image, awb_vector)
    
    # Button to reset to original image
    reset_button_ax = plt.axes([0.8, 0.5, 0.1, 0.075])
    reset_button = Button(reset_button_ax, 'Original Image')
    reset_button.on_clicked(lambda event: show_white_balance_correction(img_ax, illuminatiove_vector_ax, demosaiced_image, None))

    # Button to reset to original AWB
    reset_awb_button_ax = plt.axes([0.8, 0.35, 0.1, 0.1])
    reset_awb_button = Button(reset_awb_button_ax, 'Auto \nWhite Balance')
    reset_awb_button.on_clicked(lambda event: show_white_balance_correction(img_ax, illuminatiove_vector_ax, demosaiced_image, awb_vector))
    
    cid = fig.canvas.mpl_connect('button_press_event', partial(interactive_white_balance_correction, img_ax=img_ax, illuminatiove_vector_ax=illuminatiove_vector_ax, demosaiced_image=demosaiced_image))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <DNG file path>")
        sys.exit(1)
    filename = sys.argv[1]
    main(filename)
