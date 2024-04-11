from cython.parallel cimport prange
cimport cython
from libc.math cimport abs, sqrt
from libc.string cimport memcpy
import numpy as np
import cv2
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void align_image_cosin(float[:,:] ref_im, float[:,:] comp_im, cython.int[:,:,:] pre_alignment, cython.int search_radius, cython.int tile_size, cython.int h, cython.int w, cython.int[:,:,:] motion_vector, float[:,:] err_map) nogil:
    cdef int stride = tile_size // 2
    cdef int num_tiles_x = (h - tile_size) // stride + 1
    cdef int thread_idx
    cdef int best_dx, best_dy, i, j, cim_x, cim_y, dx, dy, x, y
    cdef float max_sim = -1, corr = 0.0, sq_sum = 0.0, sq_sum_ref = 0.0, sim = 0.0
    for thread_idx in prange(num_tiles_x):
        max_sim = -1
        i = thread_idx * stride
        for j from 0 <= j < w by stride:
            max_sim = -1
            for dx from -search_radius <= dx < search_radius by 1:
                for dy from -search_radius <= dy < search_radius by 1:
                    if pre_alignment[i, j, 0] + i + dx < 0 or pre_alignment[i, j, 0] + dx + i + tile_size >= h or pre_alignment[i, j, 1] + j + dy < 0 or pre_alignment[i, j, 1] + dy + j + tile_size >= w:
                        continue
                    corr = 0.0
                    sq_sum = 0.0
                    sq_sum_ref = 0.0
                    for x in range(tile_size):
                        for y in range(tile_size):
                            cim_x = pre_alignment[i, j, 0] + dx + x + i
                            cim_y = pre_alignment[i, j, 1] + dy + y + j
                            corr = corr + ref_im[i + x, j + y] * comp_im[cim_x, cim_y]
                            sq_sum = sq_sum + comp_im[cim_x, cim_y] * comp_im[cim_x, cim_y]
                            sq_sum_ref = sq_sum_ref + ref_im[i + x, j + y] * ref_im[i + x, j + y]
                    sim = corr / (sqrt(sq_sum) * sqrt(sq_sum_ref) + 1e-3)
                    if sim > max_sim:
                        max_sim = sim
                        best_dx = dx
                        best_dy = dy
            for x in range(stride):
                for y in range(stride):
                    motion_vector[i + x, j + y, 0] = pre_alignment[i, j, 0] + best_dx
                    motion_vector[i + x, j + y, 1] = pre_alignment[i, j, 1] + best_dy
                    err_map[i + x, j + y] = max_sim
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void align_image_l1(float[:,:] ref_im, float[:,:] comp_im, cython.int[:,:,:] pre_alignment, cython.int search_radius, cython.int tile_size, cython.int h, cython.int w, cython.int[:,:,:] motion_vector, float[:,:] err_map) nogil:
    cdef int stride = tile_size
    cdef int num_tiles_x = (h - tile_size) // stride + 1
    cdef int thread_idx
    cdef int best_dx, best_dy, i, j, cim_x, cim_y, dx, dy, x, y
    cdef float min_diff = 1000.0, diff = 0.0, ref_grad, cand_grad
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
            min_diff = min_diff / (tile_size * tile_size)

            for x in range(stride):
                for y in range(stride):
                    motion_vector[i + x, j + y, 0] = pre_alignment[i, j, 0] + best_dx
                    motion_vector[i + x, j + y, 1] = pre_alignment[i, j, 1] + best_dy
                    err_map[i + x, j + y] = min_diff
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void align_image_l2(float[:,:] ref_im, float[:,:] comp_im, cython.int[:,:,:] pre_alignment, cython.int search_radius, cython.int tile_size, cython.int h, cython.int w, cython.int[:,:,:] motion_vector, float[:,:] err_map) nogil:
    cdef int stride = tile_size // 2
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
                            diff = diff + (ref_im[i + x, j + y] - comp_im[cim_x, cim_y]) * (ref_im[i + x, j + y] - comp_im[cim_x, cim_y])
                    if diff < min_diff:
                        min_diff = diff
                        best_dx = dx
                        best_dy = dy
            for x in range(stride):
                for y in range(stride):
                    motion_vector[i + x, j + y, 0] = pre_alignment[i, j, 0] + best_dx
                    motion_vector[i + x, j + y, 1] = pre_alignment[i, j, 1] + best_dy
                    err_map[i + x, j + y] = min_diff / (tile_size * tile_size)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void align_and_merge_2im(float[:,:,:] comp_feature, float[:,:,:] ref_feature, float[:,:,:,:] comp_input, float[:,:,:,:] comp_output, cython.int[:] pyr_h, cython.int[:] pyr_w):     
    # ref_feature: level, height, width
    # comp_feature: level, height, width
    # comp_input: channel, level, height, width
    # comp_output: channel, level, height, width
    cdef int lvl, j, x, y, t, lvl_h
    cdef int num_level = comp_feature.shape[0]
    cdef int im_h = ref_feature.shape[1]
    cdef int im_w = ref_feature.shape[2]
    cdef int[:,:,:] pre_alignment = np.zeros((im_h, im_w, 2), dtype=np.int32)
    cdef int[:,:,:] motion_vector = np.zeros((im_h, im_w, 2), dtype=np.int32)
    cdef float[:,:] err_map = np.zeros((im_h, im_w), dtype=np.float32)
    cdef int err_idx = comp_output.shape[0] - 1
    for lvl in range(num_level):
        if lvl <= 0:
            align_image_l1(ref_feature[lvl], comp_feature[lvl], pre_alignment, 8, 8, pyr_h[lvl], pyr_w[lvl], motion_vector, err_map)
        elif lvl == num_level - 1:
            align_image_l1(ref_feature[lvl], comp_feature[lvl], pre_alignment, 8, 8, pyr_h[lvl], pyr_w[lvl], motion_vector, err_map)
        else:
            align_image_l1(ref_feature[lvl], comp_feature[lvl], pre_alignment, 8, 8, pyr_h[lvl], pyr_w[lvl], motion_vector, err_map)
        if lvl < (num_level - 1):
            lvl_h = pyr_h[lvl]
            for x in prange(lvl_h, nogil=True):
                for y in range(pyr_w[lvl + 1]):
                    pre_alignment[x, y, 0] = motion_vector[x // 2, y // 2, 0] * 2
                    pre_alignment[x, y, 1] = motion_vector[x // 2, y // 2, 1] * 2
        for t in prange(comp_input.shape[0], nogil=True):
            for x in range(pyr_h[lvl]):
                for y in range(pyr_w[lvl]):
                    comp_output[t, lvl, x, y] = comp_input[t, lvl, motion_vector[x, y, 0] + x, motion_vector[x, y, 1] + y]
        memcpy(&comp_output[err_idx, lvl, 0, 0], &err_map[0, 0], im_h * im_w * sizeof(float))