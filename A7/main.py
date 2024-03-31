import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pillow_lut
import argparse
from scipy.interpolate import CubicSpline
import os
import matplotlib
knot_points = {'light' : np.array([(0, 0), (78, 73), (177, 182), (255, 255)]) / 255.0,
               'medium' : np.array([(0, 0), (73, 56), (164, 164), (255, 255)]) / 255.0,
               'strong' : np.array([(0, 0), (75, 59), (150, 150), (190, 205), (255, 255)]) / 255.0,
               'none' : np.array([(0, 0), (255, 255)]) / 255.0}
def identity(x):
    return x
# Load Image
def load_image(image_path):
    return np.array(Image.open(image_path)) / 255.
# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Apply 3D LUT and tone curve to an image.')
    parser.add_argument('--image_file', type=str, help='Path to the image file')
    parser.add_argument('--cube_file', type=str, help='Path to the .CUBE file')
    parser.add_argument('--tone_choice', type=str, choices=['light', 'medium', 'strong', 'none'], help='Tone curve choice')
    return parser.parse_args()

# Main Function
def main(args):
    image = load_image(args.image_file)
    
    lut = pillow_lut.load_cube_file(args.cube_file)
    image_with_lut = np.apply_along_axis(lambda x : pillow_lut.sample_lut_linear(lut, x), axis=-1, arr=image)
    
    f = plt.figure(figsize=(15, 4))
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+0+0")
    ax = f.add_subplot(1,5,1)
    ax.imshow(image, aspect='equal', vmin=0, vmax=1)
    ax.set_title('Original Image')
    ax.axis('off')
    
    ax = f.add_subplot(1, 5, 2, projection='3d')
    lut_value = np.array(lut.table).reshape(lut.size[0], lut.size[1], lut.size[2], 3)
    r_index, g_index, b_index = np.meshgrid(np.arange(0, 1.0, 1.0 / lut.size[0]), np.arange(0, 1.0, 1.0 / lut.size[1]), np.arange(0, 1.0, 1.0 / lut.size[2]))
    ax.scatter(r_index.reshape(-1), g_index.reshape(-1), b_index.reshape(-1), c = lut_value.reshape(-1, 3))
    ax.set_title(os.path.split(os.path.basename(args.cube_file))[-1])
    ax.axis('off')

    ax = f.add_subplot(1,5,3)
    ax.imshow(image_with_lut, aspect='equal', vmin=0, vmax=1)
    ax.set_title('Applied LUT')
    ax.axis('off')
    
    if args.tone_choice == 'none':
        final_image = image_with_lut
        tone_curve_fn = identity
    else:
        tone_curve_fn = CubicSpline(knot_points[args.tone_choice][:, 0], knot_points[args.tone_choice][:, 1])
        h, w, c = image_with_lut.shape
        final_image = tone_curve_fn(image_with_lut.reshape(-1)).reshape(h, w, c)
    x = np.arange(256) / 255.
    ax = f.add_subplot(1,5,4)
    ax.plot(x, tone_curve_fn(x))
    ax.scatter(knot_points[args.tone_choice][:, 0], knot_points[args.tone_choice][:, 1], marker='o')
    ax.set_aspect('equal')
    ax.set_title(args.tone_choice)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    ax = f.add_subplot(1,5,5)
    ax.imshow(final_image, aspect='equal', vmin=0, vmax=1)
    ax.set_title('Final image')
    ax.axis('off')
    plt.show()        

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
        
