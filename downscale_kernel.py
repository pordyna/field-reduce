import numpy as np
from numba import njit, prange, generated_jit


@njit(parallel=True, inline='always')
def _downscale_1d(input_arr, output_arr):
    bin_length = input_arr.shape[0] // output_arr.shape[0]
    for ii in prange(output_arr.shape[0]):
        output_arr[ii] = np.mean(input_arr[bin_length * ii: bin_length * (ii + 1)])


@njit(parallel=True, inline='always')
def _reduce_2d_1(input_arr, output_arr):
    for ii in prange(output_arr.shape[0]):
        _downscale_1d(input_arr[ii, :], output_arr[ii, :], output_arr.shape[1])


@njit(parallel=True, inline='always')
def _reduce_2d_0(input_arr, output_arr):
    for ii in prange(output_arr.shape[1]):
        _downscale_1d(input_arr[:, ii], output_arr[:, ii], output_arr.shape[0])


@njit(parelle=True, inline='always')
def _downscale_2d(input_arr, output_arr):
    tmp_array = np.empty(input_arr.shape[0], output_arr.shape[1])
    _reduce_2d_1(input_arr, tmp_array)
    _reduce_2d_0(tmp_array, output_arr)


@njit(parelle=True, inline='always')
def _downscale_3d(input_arr, output_arr):
    tmp_array = np.empty(input_arr.shape[0], output_arr.shape[1], output_arr.shape[2])
    for ii in prange(input_arr.shape[0]):
        _downscale_2d(input_arr[ii, :, :], tmp_array[ii, :, :])
    for ii in prange(output_arr.shape[2]):
        _reduce_2d_0(tmp_array[:, :, ii], output_arr[:, :, ii])


@generated_jit(nopython=True)
def downscale(input_arr, output_arr):
    if input_arr.ndim == 1:
        return lambda a, b: _downscale_1d(a, b)
    elif input_arr.ndim == 2:
        return lambda a, b: _downscale_2d(a, b)
    elif input_arr.ndim == 3:
        return lambda a, b: _downscale_3d(a, b)
