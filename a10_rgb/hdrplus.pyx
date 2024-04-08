from cython.parallel cimport prange
cimport cython
from libc.math cimport abs
from libc.string cimport memcpy
import numpy as np
import cv2
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void align_image_l1(float[:,:] ref_im, float[:,:] comp_im, cython.int[:,:,:] pre_alignment, cython.int search_radius, cython.int tile_size, cython.int h, cython.int w, cython.int[:,:,:] motion_vector, float[:,:] err_map) nogil:
    cdef int stride = tile_size
    cdef int num_tiles_x = (h - tile_size) // stride + 1
    cdef int thread_idx
    cdef int best_dx, best_dy, i, j, cim_x, cim_y, dx, dy, x, y
    cdef float min_diff = 1000.0, diff = 0.0
    for thread_idx in prange(num_tiles_x):
        min_diff = 1000.0
        i = thread_idx * stride
        for j from 0 <= j < w by stride:
            min_diff = 1000.0
            for dx from -search_radius <= dx < search_radius by 1:
                for dy from -search_radius <= dy < search_radius by 1:
                    if pre_alignment[i, j, 0] + i + dx < 0 or pre_alignment[i, j, 0] + dx + i + tile_size >= h or pre_alignment[i, j, 1] + j + dy < 0 or pre_alignment[i, j, 1] + dy + j + tile_size >= w:
                        continue
                    diff = 0.0
                    for x in range(tile_size):
                        for y in range(tile_size):
                            cim_x = pre_alignment[i, j, 0] + dx + x + i
                            cim_y = pre_alignment[i, j, 1] + dy + y + j
                            diff = diff + abs(ref_im[i + x, j + y] - comp_im[cim_x, cim_y])
                    if diff < min_diff:
                        min_diff = diff
                        best_dx = dx
                        best_dy = dy
            for x in range(tile_size):
                for y in range(tile_size):
                    motion_vector[i + x, j + y, 0] = pre_alignment[i, j, 0] + best_dx
                    motion_vector[i + x, j + y, 1] = pre_alignment[i, j, 1] + best_dy
                    err_map[i + x, j + y] = min_diff / (tile_size * tile_size)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void upsamle_motion_vector(int[:,:,:] motion_vector, int[:,:,:] new_motion_vector, int new_h, int new_w) nogil:
    cdef int i, j
    for i in range(new_h):
        for j in range(new_w):
            new_motion_vector[i, j, 0] = motion_vector[i // 2, j // 2, 0] * 2
            new_motion_vector[i, j, 1] = motion_vector[i // 2, j // 2, 1] * 2
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef void align_and_merge(float[:,:, :,:,:] comp_im,
#                     float[:,:,:,:] ref, float[:,:,:,:] output, cython.int[:] pyr_h, cython.int[:] pyr_w):
#     # comp im: bayer_channel, im index, pyr level, height, width, 
#     # ref: pyr bayer_channel, level, height, width
#     # output equiv to ref
#     cdef int i, j, x, y
#     cdef int num_comp_im = comp_im.shape[1]
#     cdef int num_channel = comp_im.shape[0]
#     cdef im_h = ref.shape[2]
#     cdef im_w = ref.shape[3]
#     cdef int[:,:,:] pre_alignment = np.zeros((im_h, im_w, 2), dtype=np.int32)
#     cdef int[:,:,:] motion_vector = np.zeros((im_h, im_w, 2), dtype=np.int32)
#     # assume pyr level is from coarse to fine
#     for i in prange(num_comp_im, nogil=True):
#         for j in range(comp_im.shape[2]):
#             align_image_l1(ref[0, j, :,:], comp_im[0, i, j], pre_alignment, 4, 16, pyr_h[j], pyr_w[j], motion_vector)
#             for x in range(pyr_h[j]):
#                 for y in range(pyr_w[j]):
#                     output[1, j, x, y] += comp_im[1, i, j, motion_vector[x, y, 0] + x, motion_vector[x, y, 1] + y]
#                     output[2, j, x, y] += comp_im[2, i, j, motion_vector[x, y, 0] + x, motion_vector[x, y, 1] + y]
#                     output[3, j, x, y] += comp_im[3, i, j, motion_vector[x, y, 0] + x, motion_vector[x, y, 1] + y]
#                     output[4, j, x, y] += comp_im[4, i, j, motion_vector[x, y, 0] + x, motion_vector[x, y, 1] + y]
#             if j < (comp_im.shape[2] - 1):
#                 # upsamle_motion_vector(motion_vector, pre_alignment,  pyr_h[j + 1], pyr_w[j + 1])
#                 for x in range(pyr_h[j + 1]):
#                     for y in range(pyr_w[j + 1]):
#                         pre_alignment[i, j, 0] = motion_vector[i // 2, j // 2, 0] * 2
#                         pre_alignment[i, j, 1] = motion_vector[i // 2, j // 2, 1] * 2
#     for j in range(output.shape[1]):
#         for x in range(pyr_h[j]):
#             for y in range(pyr_w[j]):
#                     output[1, j, x, y] = output[1, j, x, y] / (num_comp_im + 1)
#                     output[2, j, x, y] = output[2, j, x, y] / (num_comp_im + 1)
#                     output[3, j, x, y] = output[3, j, x, y] / (num_comp_im + 1)
#                     output[4, j, x, y] = output[4, j, x, y] / (num_comp_im + 1)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void align_and_merge_2im(float[:,:,:] comp_feature, float[:,:,:] ref_feature, float[:,:,:,:] comp_input, float[:,:,:,:] comp_output, cython.int[:] pyr_h, cython.int[:] pyr_w):     
    # ref_feature: level, height, width
    # comp_feature: level, height, width
    # comp_input: channel, level, height, width
    # comp_output: channel, level, height, width
    cdef int lvl, j, x, y, t
    cdef int num_level = comp_feature.shape[0]
    cdef int im_h = ref_feature.shape[1]
    cdef int im_w = ref_feature.shape[2]
    cdef int[:,:,:] pre_alignment = np.zeros((im_h, im_w, 2), dtype=np.int32)
    cdef int[:,:,:] motion_vector = np.zeros((im_h, im_w, 2), dtype=np.int32)
    cdef float[:,:] err_map = np.zeros((im_h, im_w), dtype=np.float32)
    cdef int err_idx = comp_output.shape[0] - 1
    for lvl in range(num_level):
        align_image_l1(ref_feature[lvl], comp_feature[lvl], pre_alignment, 8, 8, pyr_h[lvl], pyr_w[lvl], motion_vector, err_map)
        if lvl < (num_level - 1):
            for x in range(pyr_h[lvl + 1]):
                for y in range(pyr_w[lvl + 1]):
                    pre_alignment[x, y, 0] = motion_vector[x // 2, y // 2, 0] * 2
                    pre_alignment[x, y, 1] = motion_vector[x // 2, y // 2, 1] * 2
        for t in prange(comp_input.shape[0], nogil=True):
            for x in range(pyr_h[lvl]):
                for y in range(pyr_w[lvl]):
                    comp_output[t, lvl, x, y] = comp_input[t, lvl, motion_vector[x, y, 0] + x, motion_vector[x, y, 1] + y]
        memcpy(&comp_output[err_idx, lvl, 0, 0], &err_map[0, 0], im_h * im_w * sizeof(float))