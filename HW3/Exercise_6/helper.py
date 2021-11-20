import math
from scipy import signal
import cv2
import numpy as np


def normalize_image(image):
    image = image + 20
    new_img = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (np.linalg.norm(image[i, j]) < 120):
                new_img[i, j] = image[i, j]
    new_img = np.array(new_img, dtype='uint8')

    return new_img


def get_dots(height, width, dots_count):
    center_x = width // 2 - 150
    center_y = height // 2 - 50
    radius = 250

    # these lines copy from Internet
    teta = np.linspace(0, 2 * np.pi, dots_count)
    sin = center_x + radius * np.sin(teta)
    cos = center_y + radius * np.cos(teta)
    dots = np.array([sin, cos], dtype='int').T

    return dots, center_x, center_y, radius


def remove_out_of_circle(image, center_x, center_y, radius):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if math.sqrt((i - center_y) ** 2 + (j - center_x) ** 2) > radius:
                image[i, j] = 0

    return image


def gaussian_derivative(x, y, mu, sigma):
    return ((-(x - mu)) / (2 * np.pi * sigma ** 4)) * np.exp(-(((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma ** 2)))


def gaussian_derivative_filter(size, sigma, axis):
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


def get_gradients(image):
    gradient_x = signal.convolve2d(image, gaussian_derivative_filter(7, 1, 'x'))
    gradient_y = signal.convolve2d(image, gaussian_derivative_filter(7, 1, 'y'))

    return gradient_x, gradient_y


def move_one_point(point_index, dots, d, alpha, beta, gamma, height, width, dots_count, gradient_x, gradient_y):
    v = dots[point_index]
    x, y = v
    minimum = 1e9
    best_i, best_j = 0, 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_y = y + j
            new_x = x + i
            if (new_x < width and new_y < height):
                next_move_v = [new_y, new_x]
                next_move_v = np.array(next_move_v)

                next_point = point_index + 1
                previous_point = point_index - 1
                if next_point >= dots_count:
                    next_point = next_point % dots_count
                if previous_point >= dots_count:
                    previous_point = previous_point % dots_count

                next_v = dots[next_point]
                previous_v = dots[previous_point]

                internal_E1, internal_E2 = calculate_internal_E(v, next_move_v, next_v, previous_v, d)

                E_external = calculate_external_E(gradient_x, gradient_y, x, y, new_x, new_y)

                if (alpha * internal_E1) + (beta * internal_E2) + (gamma * E_external) < minimum:
                    minimum = alpha * internal_E1 + beta * internal_E2 + gamma * E_external
                    best_i, best_j = i, j
    dots[point_index] = [x + best_i, y + best_j]
    return dots


def calculate_internal_E(v, next_move_v, next_v, previous_v, d):
    internal_E1 = (
            (np.linalg.norm(next_move_v - next_v) - d + 100) ** 2 - (
            np.linalg.norm(v - next_v) - d + 100) ** 2)
    internal_E2 = (np.linalg.norm(next_move_v + next_move_v - next_v - previous_v) - np.linalg.norm(
        v + v - next_v - previous_v))
    return internal_E1, internal_E2


def calculate_external_E(gradient_x, gradient_y, x, y, new_x, new_y):
    return (-(gradient_x[new_y, new_x] ** 2 + gradient_y[new_y, new_x] ** 2) + (
            gradient_x[y, x] ** 2 + gradient_y[y, x] ** 2))


def calculate_d(dots, dots_count):
    sum = 0
    for point_index in range(dots_count):
        point = dots[point_index]
        next_point = dots[(point_index + 1) % dots_count]
        sum += np.linalg.norm(point - next_point)
    return sum / dots_count


def calculate_move(dots, dots_count, height, width, alpha, beta, gamma, gradient_x, gradient_y):
    for i in range(dots_count):
        dots = move_one_point(i, dots, calculate_d(dots, dots_count), alpha, beta, gamma, height, width, dots_count,
                              gradient_x, gradient_y)

    return dots


def make_video(size):
    # Copy from Internet
    img_array = []
    for i in range(0, 250):
        img = cv2.imread("frames/" + str(i) + ".jpg")
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('out/movie01.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
