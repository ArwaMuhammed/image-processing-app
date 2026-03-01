import numpy as np

def apply_filter(image, filter_type):
    if filter_type == "Average (3x3)":
        return average_filter(image)

    elif filter_type == "Gaussian (3x3)":
        return gaussian_filter(image)

    elif filter_type == "Median (3x3)":
        return median_filter(image)

    return image


def average_filter(image):
    kernel = np.ones((3, 3)) / 9
    return apply_convolution(image, kernel)

def gaussian_filter(image):
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16
    return apply_convolution(image, kernel)

def median_filter(image):
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                region = padded[i:i+3, j:j+3, c]
                output[i, j, c] = np.median(region)

    return output


def apply_convolution(image, kernel):
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                region = padded[i:i+3, j:j+3, c]
                output[i, j, c] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)