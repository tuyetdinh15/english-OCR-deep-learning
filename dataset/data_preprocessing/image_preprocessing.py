from pathlib import Path

import cv2


def read_image(path: str | Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def binarization(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)


def dilate(image, words: bool = False):
    height, width, iterations = (6, 6, 3) if words else (3, 1, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
    return cv2.dilate(image.copy(), kernel, iterations=iterations)


def find_rectangles(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectangles = [list(cv2.boundingRect(contour)) for contour in contours]
    return sorted(rectangles, key=lambda rectangle: rectangle[0])
