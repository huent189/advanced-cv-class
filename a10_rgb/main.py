import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from hdrplus import align_and_merge_2im
import math
NUM_LEVELS = 4
ERR_THRESHOLD = 0.15

def ldr_to_hdr(img, expo, gamma=2.2):
    img = img.clip(0, 1)
    img = np.power(img, gamma) # linearize
    img /= expo
    return img
def compute_quality_map(image, sigma=0.2):
    gray = np.dot(image, [0.2989, 0.5870, 0.1140])
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    constrast = scipy.ndimage.convolve(gray, laplacian_filter, mode='mirror')
    constrast = np.abs(constrast)
    saturation = np.std(image, axis=2)
    well_exposedness = np.exp(-((image - 0.5) ** 2) / (2 * (sigma ** 2)))
    well_exposedness = np.prod(well_exposedness, axis=2)
    quality_map = constrast * saturation * well_exposedness + 1e-8
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
def process_input_image(inputs):
    path, expo = inputs
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 255.0
    im = im.astype(np.float32)
    hdr_im = ldr_to_hdr(im, expo)
    hdr_gray = np.dot(hdr_im, [0.2989, 0.5870, 0.1140])
    quality_map = compute_quality_map(im)

    pyr_hdr = compute_gaussian_pyramid(hdr_gray, NUM_LEVELS)
    pyr_quality = compute_gaussian_pyramid(quality_map, NUM_LEVELS)
    pyr_rgb = compute_laplacian_pyramid(im, NUM_LEVELS)
    # pyr_hdr.reverse()
    # pyr_quality.reverse()
    # pyr_rgb.reverse()
    return im, pyr_hdr, pyr_quality, pyr_rgb

def exposure_fusion(laplacian_pyramids, quality_map_pyramids, get_pyr = False):
    blended_pyramids = []
    for i in range(NUM_LEVELS):
        blended = 0
        sum_weights = 0
        for j in range(len(laplacian_pyramids)):
            blended += laplacian_pyramids[j][i] * quality_map_pyramids[j][i][:,:,None]
            sum_weights += quality_map_pyramids[j][i][:,:,None]
        blended_pyramids.append(blended / sum_weights)
    output = reconstruct_from_laplacian_pyramid(blended_pyramids)
    output = np.clip(output, 0, 1)
    if get_pyr:
        return output, blended_pyramids
    return output

def exposure_fusion_v2(laplacian_pyramids, quality_map_pyramids, max_level):
    blended_pyramids = []
    new_im_pyr = [pyr[:-1] + compute_laplacian_pyramid(pyr[-1], max_level - NUM_LEVELS+1) for pyr in laplacian_pyramids]
    new_quality_pyr = [quality_map_pyr[:-1] + compute_gaussian_pyramid(quality_map_pyr[-1], max_level - NUM_LEVELS+1) for quality_map_pyr in quality_map_pyramids]
    for i in range(max_level):
        blended = 0
        sum_weights = 0
        for j in range(len(laplacian_pyramids)):
            
            blended += new_im_pyr[j][i] * new_quality_pyr[j][i][:,:,None]
            sum_weights += new_quality_pyr[j][i][:,:,None]
        blended_pyramids.append(blended / sum_weights)
    output = reconstruct_from_laplacian_pyramid(blended_pyramids)
    output = np.clip(output, 0, 1)
    return output

def pack(pyr):
    pyr.reverse()
    last_lvl = pyr[-1]
    
    num_pyr = len(pyr)
    output = np.zeros([num_pyr] + list(last_lvl.shape), dtype=np.float32)
    for i in range(num_pyr):
        h, w = pyr[i].shape[:2]
        output[i, :h, :w] = pyr[i]
    pyr.reverse()
    return output
def unpack(packed, pyr_h, pyr_w):
    outputs = []
    for lvl in range(packed.shape[0]):
        h = pyr_h[lvl]
        w = pyr_w[lvl]
        outputs.append(packed[lvl, :h, :w])
    outputs.reverse()
    return outputs
def display_output(ev0, output_no_align, output_align):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(ev0)
    axs[0].set_title('EV 0')
    axs[0].set_axis_off()
    axs[0].set_aspect('equal')
    axs[1].imshow(output_no_align)
    axs[1].set_title('output_no_align Image')
    axs[1].set_axis_off()
    axs[1].set_aspect('equal')
    axs[2].imshow(output_align)
    axs[2].set_title('output_align Image')
    axs[2].set_axis_off()
    axs[2].set_aspect('equal')
    plt.show()

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
                # im = im / vmax
                im_fig = ax.imshow(im, cmap='rainbow', vmin=0, vmax=ERR_THRESHOLD)
                # ax.colorbar()
                # plt.colorbar(im_fig, ax=ax)
            else:
                ax.imshow(im)
            ax.set_title(f'Level {col}')
            ax.set_axis_off()
            ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    import sys,glob, time
    from multiprocessing import Pool
    burst_folder = sys.argv[1]
    burst_img_paths = sorted(glob.glob(burst_folder + '*.*'))
    expos = [1, 2**2, 2**4]
    t0 = time.time()
    with Pool(len(burst_img_paths)) as p:
        results = p.map(process_input_image, zip(burst_img_paths, expos))
    t1 = time.time()
    print(f"Processing {len(burst_img_paths)} images took {t1 - t0} seconds")
    pyr_rgbs = [r[3] for r in results]
    pyr_qualities = [r[2] for r in results]
    im_h, im_w = results[0][0].shape[:2]
    output_no_align = exposure_fusion_v2(pyr_rgbs, pyr_qualities, min(int(math.log2(im_h)), int(math.log2(im_w))))
    ref_index = 0
    ref_pyr_hdr = results[ref_index][1]
    pyr_h = []
    pyr_w = []
    for i in range(NUM_LEVELS):
        h, w = ref_pyr_hdr[i].shape[:2]
        pyr_h.append(h)
        pyr_w.append(w)
    pyr_h.reverse()
    pyr_w.reverse()
    pyr_h = np.array(pyr_h, dtype=np.int32)
    pyr_w = np.array(pyr_w, dtype=np.int32)
    ref_feature = pack(ref_pyr_hdr)
    err_pyrs = []
    for comp_index in range(1, len(results)):
        comp_feature = pack(results[comp_index][1])
        comp_rgb = pack(results[comp_index][3])
        comp_quality_map = pack(results[comp_index][2])
        comp_input = np.concatenate([comp_rgb, comp_quality_map[..., None]], axis=-1).transpose(3, 0, 1, 2)
        comp_output = np.zeros((comp_input.shape[0]+1, comp_input.shape[1], comp_input.shape[2], comp_input.shape[-1]), dtype=np.float32)
        t0 = time.time()
        align_and_merge_2im(comp_feature, ref_feature, comp_input, comp_output, pyr_h, pyr_w)
        t1 = time.time()
        print ('Parallel Time: %f' % (t1-t0))
        aligned_rgb = comp_output[:3].transpose(1, 2, 3, 0)
        aligned_quality = comp_output[3] * (comp_output[-1] < ERR_THRESHOLD).astype(np.float32)
        aligned_rgb_pyr = unpack(aligned_rgb, pyr_h, pyr_w)
        err_pyr = unpack(comp_output[-1], pyr_h, pyr_w)
        err_pyrs.append(err_pyr)
        aligned_quality_pyr = unpack(aligned_quality, pyr_h, pyr_w)
        pyr_rgbs[comp_index] = aligned_rgb_pyr
        pyr_qualities[comp_index] = aligned_quality_pyr

    im_h, im_w = results[0][0].shape[:2]
    output_align = exposure_fusion_v2(pyr_rgbs, pyr_qualities, min(int(math.log2(im_h)), int(math.log2(im_w))))
    display_output(results[0][0], output_no_align, output_align)
    print(err_pyrs[0][0].shape)
    display_pyramid(err_pyrs, [f'Error Map 1', f'Error Map 2'])

    # fig, ax = plt.subplots(1, 3)
    # for i in range(3):
        # im_fig = ax[i].imshow(np.abs(results[i][0] - results[0][0]), cmap='rainbow')
        # plt.colorbar(im_fig, ax=ax[i])
        # ax[i].imshow(results[i][0])
    # plt.show()