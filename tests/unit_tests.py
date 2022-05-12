import numpy as np
import pytest
import downscale_kernel


def test_1d_even():
    size = 126
    bin_size = 2
    large = np.random.random(size)
    output = np.empty(126 // bin_size)
    downscale_kernel._downscale_1d(large, output)
    reference = (large[::2] + large[1::2]) / 2
    assert np.all(np.isclose(reference, output))


def test_1d_odd():
    size = 126
    bin_size = 3
    large = np.random.random(size)
    output = np.empty(126 // bin_size)
    downscale_kernel._downscale_1d(large, output)
    reference = (large[::3] + large[1::3] + large[2::3]) / 3
    assert np.all(np.isclose(reference, output))


def test_2d():
    shape = (126, 90)
    bin_size = (3, 2)
    large = np.random.random(size=shape[0] * shape[1]).reshape(shape)

    reference_0 = (large[0::bin_size[0], :] + large[1::bin_size[0], :] + large[2::bin_size[0], :])
    output_0 = np.empty((shape[0]//bin_size[0], shape[1]))
    downscale_kernel._reduce_2d_0(large, output_0)
    assert np.all(np.isclose(output_0, reference_0))

    reference_1 = (large[:, 0::bin_size[1]] + large[:, 1::bin_size[1]])
    output_1 = np.empty((shape[0], shape[1]//bin_size[1]))
    downscale_kernel._reduce_2d_1(large, output_1)
    assert np.all(np.isclose(output_1, reference_1))

    reference = (large[::3, ::2] + large[1::3, ::2] + large[2::3, ::2]
                 + large[::3, 1::2] + large[1::3, 1::2] + large[2::3, 1::2]) / 6
    output = np.empty((shape[0] // bin_size[0], shape[1] // bin_size[1]))
    downscale_kernel._downscale_2d(large, output)
    assert np.all(np.isclose(reference, output))


def test_3d():
    shape = (126, 90, 20)
    bin_size = (2, 3, 2)
    large = np.empty(shape)
    large = np.random.random(large.size).reshape(shape)
    reference = (large[::2, ::3, ::2] + large[::2, 1::3, ::2] + large[::2, 2::3, ::2]
                 + large[1::2, ::3, ::2] + large[1::2, 1::3, ::2] + large[1::2, 2::3, ::2]
                 + large[1::2, ::3, 1::2] + large[1::2, 1::3, 1::2] + large[1::2, 2::3, 1::2]
                 + large[::2, ::3, 1::2] + large[::2, 1::3, 1::2] + large[::2, 2::3, 1::2]) / 12
    output = np.empty((shape[0]//bin_size[0], shape[1]//bin_size[1], shape[2]//bin_size[2]))
    downscale_kernel._downscale_3d(large, output)
    assert np.all(np.isclose(reference, output))

