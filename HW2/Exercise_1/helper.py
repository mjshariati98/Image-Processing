import cv2
import math
import numpy as np


def gaussian_derivative(x, y, mu, sigma):
    return ((-(x - mu)) / (2 * math.pi * sigma ** 4)) * math.exp(-(((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma ** 2)))


def gaussian_derivative_filter(size, sigma, axis):
    """
    build a size*size filter of derivative of gaussian function and return it.

    :param size: size of filter(matrix)
    :param sigma: the sigma of gaussian function
    :param axis: the x-coordinate or y-coordinate
    :return: filter
    """
    mu = size // 2
    mat = np.zeros((size, size))
    if axis == 'x':
        for i in range(size):
            for j in range(size):
                mat[i, j] = gaussian_derivative(j, i, mu, sigma)
    else:
        for i in range(size):
            for j in range(size):
                mat[i, j] = gaussian_derivative(i, j, mu, sigma)

    return mat


def row_gaussian_derivative_in_x_axis(x, mu, sigma):
    return (-(x - mu) / (math.sqrt(2 * math.pi) * sigma ** 2)) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def col_gaussian_derivative_in_x_axis(y, mu, sigma):
    return (1 / (math.sqrt(2 * math.pi) * sigma ** 2)) * math.exp(-((y - mu) ** 2) / (2 * sigma ** 2))


def row_gaussian_derivative_in_y_axis(x, mu, sigma):
    return (1 / (math.sqrt(2 * math.pi) * sigma ** 2)) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def col_gaussian_derivative_in_y_axis(y, mu, sigma):
    return (-(y - mu) / (math.sqrt(2 * math.pi) * sigma ** 2)) * math.exp(-((y - mu) ** 2) / (2 * sigma ** 2))


def row_gaussian_derivative_in_x_axis_filter(size, sigma):
    mu = size // 2
    mat = np.zeros((1, size))
    for i in range(size):
        mat[0, i] = row_gaussian_derivative_in_x_axis(i, mu, sigma)

    return mat


def col_gaussian_derivative_in_x_axis_filter(size, sigma):
    mu = size // 2
    mat = np.zeros((size, 1))
    for i in range(size):
        mat[i, 0] = col_gaussian_derivative_in_x_axis(i, mu, sigma)

    return mat


def row_gaussian_derivative_in_y_axis_filter(size, sigma):
    mu = size // 2
    mat = np.zeros((1, size))
    for i in range(size):
        mat[0, i] = row_gaussian_derivative_in_y_axis(i, mu, sigma)

    return mat


def col_gaussian_derivative_in_y_axis_filter(size, sigma):
    mu = size // 2
    mat = np.zeros((size, 1))
    for i in range(size):
        mat[i, 0] = col_gaussian_derivative_in_y_axis(i, mu, sigma)

    return mat


def get_distributed_image(image):
    distributed_image = image / image.max() * 255
    distributed_image = np.asarray(distributed_image, dtype='uint8')
    return distributed_image


def check_equality(matrix_1, matrix_2):
    total_elements = 0
    equal_element = 0
    for i in range(matrix_1.shape[0]):
        for j in range(matrix_2.shape[1]):
            total_elements += 1
            if abs(matrix_1[i, j] - matrix_2[i, j]) < 0.01:
                equal_element += 1

    return total_elements == equal_element


def clean(grad, theta):
    threshold_image = grad

    for i in range(1, theta.shape[0] - 1):
        for j in range(1, theta.shape[1] - 1):
            if 0 <= math.fabs(theta[i, j]) <= (math.pi / 8) or (7 * math.pi / 8) <= math.fabs(theta[i, j]) <= math.pi:
                if grad[i, j + 1] > grad[i, j] or grad[i, j - 1] > grad[i, j]:
                    threshold_image[i, j] = 0
            elif (math.pi / 8) < math.fabs(theta[i, j]) <= (3 * math.pi / 8):
                if grad[i - 1, j + 1] > grad[i, j] or grad[i + 1, j - 1] > grad[i, j]:
                    threshold_image[i, j] = 0
            elif (3 * math.pi / 8) < math.fabs(theta[i, j]) <= (5 * math.pi / 8):
                if grad[i - 1, j] > grad[i, j] or grad[i + 1, j] > grad[i, j]:
                    threshold_image[i, j] = 0
            elif (5 * math.pi / 8) < math.fabs(theta[i, j]) <= (7 * math.pi / 8):
                if grad[i - 1, j - 1] > grad[i, j] or grad[i + 1, j + 1] > grad[i, j]:
                    threshold_image[i, j] = 0

    return threshold_image

