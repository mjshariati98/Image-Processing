import math
import cv2
import numpy as np
from sklearn.cluster import KMeans, MeanShift


def get_points(file):
    file_lines = file.readlines()
    x = []
    y = []
    for i in range(1, len(file_lines)):
        xx, yy = file_lines[i].split(" ")
        x.append(math.floor(float(xx) * 100))
        y.append(math.floor(float(yy) * 100))

    min_x = 0
    max_x = 0
    for i in x:
        if i > max_x:
            max_x = i
        if i < min_x:
            min_x = i

    min_y = 0
    max_y = 0
    for j in y:
        if j > max_y:
            max_y = j
        if j < min_y:
            min_y = j

    for i in range(len(x)):
        x[i] += -min_x

    for j in range(len(y)):
        y[j] += -min_y

    return len(x), x, y, min_x, max_x, min_y, max_y


def create_image_from_points(x, y, min_x, max_x, min_y, max_y):
    image = np.zeros((max_y - min_y + 1, max_x - min_x + 1))
    for i in range(len(x)):
        image[y[i], x[i]] = 255

    return image


def get_array_points(x, y):
    list = []
    for i in range(len(x)):
        list.append([x[i], y[i]])
    points = np.array(list)

    return points


def k_means(points ,x, y, width, height, file_name):
    kmeans = KMeans(n_clusters=2, random_state=1).fit(points)

    kmeans_img = np.zeros((height, width, 3))
    kmeans_img = np.vectorize(lambda x: 255)(kmeans_img)
    for i in range(len(points)):
        if kmeans.labels_[i] == 1:
            kmeans_img[y[i] - 3:y[i] + 3, x[i] - 3: x[i] + 3, 1:3] = 0

        else:
            kmeans_img[y[i] - 3:y[i] + 3, x[i] - 3: x[i] + 3, 0:2] = 0

    cv2.imwrite("out/" + file_name + ".jpg", kmeans_img)


def mean_shift(x, y, width, height, file_name):
    points = get_array_points(x, y)
    mean_shift = MeanShift(bandwidth=460).fit(points)

    mean_shift_img = np.zeros((height, width, 3))
    mean_shift_img = np.vectorize(lambda x: 255)(mean_shift_img)
    for i in range(len(points)):
        if mean_shift.labels_[i] == 1:
            mean_shift_img[y[i] - 3:y[i] + 3, x[i] - 3: x[i] + 3, 1:3] = 0

        else:
            mean_shift_img[y[i] - 3:y[i] + 3, x[i] - 3: x[i] + 3, 0:2] = 0

    cv2.imwrite("out/"+ file_name + ".jpg", mean_shift_img)


def get_points_in_polar_coordinates(file):
    file = open("../resources/Points.txt", "r")
    file_lines = file.readlines()
    r_array = []
    theta_array = []
    for i in range(1, len(file_lines)):
        xx, yy = file_lines[i].split(" ")
        point_r = math.floor(math.sqrt(float(xx) ** 2 + float(yy) ** 2) * 100)
        r_array.append(point_r)
        point_theta = math.floor(math.atan2(float(yy), float(xx)) * 100)
        theta_array.append(point_theta)

    min_r = 10000
    max_r = 0
    for i in r_array:
        if i > max_r:
            max_r = i
        if i < min_r:
            min_r = i

    min_theta = 10000
    max_theta = 0
    for j in theta_array:
        if j > max_theta:
            max_theta = j
        if j < min_theta:
            min_theta = j

    for j in range(len(theta_array)):
        theta_array[j] += -min_theta

    return r_array, theta_array, min_r, max_r, min_theta, max_theta


def create_image_from_polar_points(r, theta, max_r, min_theta, max_theta):
    image = np.zeros((max_theta - min_theta + 1, max_r + 1))
    for i in range(len(r)):
        image[theta[i], r[i]] = 255

    return image
