import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.ndimage
def display_pyramid(pyramids, subfig_titles):
    num_levels = len(pyramids[0])
    num_imgs = len(pyramids)
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=num_imgs, ncols=1)
    if num_imgs == 1:
        subfigs = [subfigs]
    vmax = max([pyr.max() for pyr in pyramids for pyr in pyr])
    for row in range(num_imgs):
        subfig = subfigs[row]
        subfig.suptitle(subfig_titles[row])
        axs = subfig.subplots(nrows=1, ncols=num_levels)
        for col, ax in enumerate(axs):
            im = pyramids[row][col]
            if len(im.shape) == 2:
                im = im / vmax
                ax.imshow(im, cmap='gray', vmin=0, vmax=vmax)
            else:
                ax.imshow(im)
            ax.set_title(f'Level {col}')
            ax.set_axis_off()
            ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
def display_output(ev0, fused_img):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(ev0)
    axs[0].set_title('EV 0')
    axs[0].set_axis_off()
    axs[0].set_aspect('equal')
    axs[1].imshow(fused_img)
    axs[1].set_title('Fused Image')
    axs[1].set_axis_off()
    axs[1].set_aspect('equal')
    plt.show()
def compute_quality_map(image, sigma=0.2):
    gray = np.dot(image, [0.2989, 0.5870, 0.1140])
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    constrast = scipy.ndimage.convolve(gray, laplacian_filter, mode='mirror')
    constrast = np.abs(constrast)
    saturation = np.std(image, axis=2)
    well_exposedness = np.exp(-((image - 0.5) ** 2) / (2 * (sigma ** 2)))
    well_exposedness = np.prod(well_exposedness, axis=2)
    quality_map = constrast * saturation * well_exposedness + 1e-5
    return quality_map

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
def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
    num_levels = len(laplacian_pyramid)
    reconstructed_image = laplacian_pyramid[-1]
    for i in range(num_levels - 2, -1, -1):
        dstsize = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        upsized_image = cv2.pyrUp(reconstructed_image, dstsize=dstsize)

        reconstructed_image = cv2.add(upsized_image, laplacian_pyramid[i])
    return reconstructed_image
def main(image_paths, num_levels, boost_factor):
    images = [cv2.imread(path)[:,:, ::-1] for path in image_paths]
    images = [image.astype(np.float32) / 255 for image in images]
    quality_maps = [compute_quality_map(image) for image in images]
    norm_factor = sum(quality_maps)
    quality_maps = [quality_map / norm_factor for quality_map in quality_maps]
    quality_map_pyramids = [compute_gaussian_pyramid(map, num_levels) for map in quality_maps]
    display_pyramid(quality_map_pyramids, [f'Gaussian Pyramid weight map for EV {i}' for i in [-1, 0, 1]])
    
    laplacian_pyramids = [compute_laplacian_pyramid(image, num_levels) for image in images]
    weights = [1] * num_levels
    weights[0] = boost_factor
    weights[1] = boost_factor
    blended_pyramids = []
    for i in range(num_levels):
        blended = 0
        for j in range(len(images)):
            blended += laplacian_pyramids[j][i] * quality_map_pyramids[j][i][:,:,None]
        blended_pyramids.append(blended * weights[i])
    display_pyramid([blended_pyramids], ['Blended Laplacian Pyramid'])
    output = reconstruct_from_laplacian_pyramid(blended_pyramids)
    output = np.clip(output, 0, 1)
    display_output(images[1], output)


