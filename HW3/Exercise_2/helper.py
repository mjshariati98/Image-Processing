import numpy as np
import cv2


def build_RGBXY_array(img, height, width):
    points = np.zeros((height, width, 5), dtype='uint8')
    points[:, :, 0:3] = img
    for i in range(height):
        for j in range(width):
            points[i, j, 3] = (j / width) * 255
            points[i, j, 4] = (i / height) * 255

    return points.reshape((height * width, 5))


def show_clusters(points, labels, width, height):
    labels_count = get_number_of_labels(labels)

    clusters = create_clusters(labels, labels_count, points)

    average_clusters_color = []
    for cluster in clusters:
        average_clusters_color.append(calculate_average_color_of_cluster(cluster))

    labels = np.reshape(labels, (height, width))
    result = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            label = labels[i, j]
            result[i, j] = average_clusters_color[label]

    cv2.imwrite("out/im05.jpg", result)


def get_number_of_labels(labels):
    labels_count = 0
    for i in labels:
        if i > labels_count:
            labels_count = i

    return labels_count


def create_clusters(labels, labels_count, points):
    clusters = []
    for i in range(labels_count + 1):
        clusters.append([])

    for i in range(len(labels)):
        clusters[labels[i]].append(points[i])
    return clusters


def calculate_average_color_of_cluster(cluster):
    cluster_size = len(cluster)
    B_avg, G_avg, R_avg = 0, 0, 0

    for point in cluster:
        B_avg += point[0]
        G_avg += point[1]
        R_avg += point[2]

    B_avg /= cluster_size
    G_avg /= cluster_size
    R_avg /= cluster_size

    return [B_avg, G_avg, R_avg]
