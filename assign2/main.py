import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import colour

def display_rgb_image(ax, image, title):
    ax.imshow(image, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])  # remove x ticks
    ax.set_yticks([])  # remove y ticks

def convert_rgb_to_lab(rgb_img, colorspace):
    # Convert RGB to XYZ
    xyz_img = colour.RGB_to_XYZ(rgb_img, colorspace, apply_cctf_decoding=True)
    # Convert XYZ to Lab
    lab_img = colour.XYZ_to_Lab(xyz_img)
    return lab_img

def plot_ab_coordinates(ax, lab_img, ab_color):
    a_channel = lab_img[:, :, 1]
    b_channel = lab_img[:, :, 2]
    ax.scatter(a_channel.flatten(), b_channel.flatten(), c=ab_color, marker='D', s=3)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_title('ab coordinates')
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-100, 0, 100])
    ax.set_aspect('equal')
    ax.grid(True)

def process_image(rgb_img, colorspace, image_title, ab_color, sub_ax):
    display_rgb_image(sub_ax[0], rgb_img, image_title)

    # Convert RGB to Lab and plot ab coordinates
    lab_img = convert_rgb_to_lab(rgb_img, colorspace)
    plot_ab_coordinates(sub_ax[1], lab_img, ab_color)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Color Space Conversion and L*ab plotting.')
    parser.add_argument('image_path_prefix', type=str, help='Path to image, excluding the color format and extension.')
    args = parser.parse_args()

    # Load images and convert to RGB
    image_path_prefix = args.image_path_prefix
    image_formats = ['ProPhoto', 'AdobeRGB', 'sRGB']
    colorspaces = [colour.models.RGB_COLOURSPACE_PROPHOTO_RGB, colour.models.RGB_COLOURSPACE_ADOBE_RGB1998, colour.models.RGB_COLOURSPACE_sRGB]
    rgb_images = [cv2.imread(f'{image_path_prefix}_{img}.png', cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0 for img in image_formats]

    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i, (rgb, cs, img_format) in enumerate(zip(rgb_images, colorspaces, image_formats)):
        process_image(rgb, cs, f'{img_format} Image', 'rgb'[i], axs[i])
    plt.show()

if __name__ == '__main__':
    main()
