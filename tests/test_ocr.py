import numpy as np

from dataset.data_preprocessing.image_preprocessing import binarization, convert_to_gray, dilate, find_rectangles


def test_convert_to_gray():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert convert_to_gray(image).shape == (8, 8)


def test_binarization_and_dilation():
    image = np.zeros((20, 20), dtype=np.uint8)
    image[8:12, 8:12] = 255
    _, binary = binarization(image)
    assert dilate(binary).shape == image.shape


def test_find_rectangles_orders_by_x_coordinate():
    image = np.zeros((30, 40), dtype=np.uint8)
    image[5:10, 25:30] = 255
    image[5:10, 5:10] = 255
    rectangles = find_rectangles(image)
    assert rectangles[0][0] < rectangles[1][0]
