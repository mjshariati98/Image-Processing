import numpy as np
import cv2
from sklearn.cluster import KMeans


def get_edge_points(edge_image):
    image_height, image_width = edge_image.shape[:2]

    points = []
    for i in range(image_height):
        for j in range(image_width):
            if edge_image[i, j] == 255:
                points.append([j, i])
    return np.array(points)


def k_means(points, number_of_clusters):
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=1).fit(points)

    cluster1_points = []
    cluster2_points = []
    cluster3_points = []
    for i in range(len(points)):
        if kmeans.labels_[i] == 0:
            cluster1_points.append(points[i])
        elif kmeans.labels_[i] == 1:
            cluster2_points.append(points[i])
        else:
            cluster3_points.append(points[i])

    cluster1_points = np.array(cluster1_points)
    cluster2_points = np.array(cluster2_points)
    cluster3_points = np.array(cluster3_points)

    clusters = [cluster1_points, cluster2_points, cluster3_points]
    return kmeans.cluster_centers_, clusters


def get_rotation_matrix(theta):
    return np.array([
        [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
        [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]
    ])


def fill_a_b_theta_table(cluster_points, table, center_x, center_y, A, AA, B, BB, ):
    for a in range(A, AA):
        for b in range(B, BB):
            for theta in range(0, 180, 10):
                for point in cluster_points:
                    point = point - np.array([center_x, center_y])
                    rotation_point = np.dot(get_rotation_matrix(-theta), point)
                    if int(np.abs(rotation_point[0])) == a or int(np.abs(rotation_point[1])) == b:
                        table[a - A, b - B, theta // 10] = table[a - A, b - B, theta // 10] + 1
    return table


def get_rectangle_width_height_theta(table, A, AA, B, BB, max):
    best_a, best_b, best_theta = 0, 0, 0
    for a in range(AA - A):
        for b in range(BB - B):
            for theta in range(18):
                if table[a, b, theta] == max:
                    best_a, best_b, best_theta = a, b, theta

    rectangle_width = best_a + A
    rectangle_height = best_b + B
    rectangle_theta = best_theta * 10

    return rectangle_width, rectangle_height, rectangle_theta


def draw_rectangle(x, y, width, height, theta, image):
    x_1 = int(x + (- height * np.sin(np.deg2rad(theta))) + (width * np.cos(np.deg2rad(theta))))  # top right corner
    y_1 = int(y + (height * np.cos(np.deg2rad(theta))) + (width * np.sin(np.deg2rad(theta))))

    x_2 = int(x + (height * np.sin(np.deg2rad(theta))) + (width * np.cos(np.deg2rad(theta))))  # bottom right
    y_2 = int(y + (- height * np.cos(np.deg2rad(theta))) + (width * np.sin(np.deg2rad(theta))))

    x_3 = int(x + (height * np.sin(np.deg2rad(theta))) - (width * np.cos(np.deg2rad(theta))))  # bottom left corner
    y_3 = int(y + (- height * np.cos(np.deg2rad(theta))) - (width * np.sin(np.deg2rad(theta))))

    x_4 = int(x + (- height * np.sin(np.deg2rad(theta))) - (width * np.cos(np.deg2rad(theta))))  # top left corner
    y_4 = int(y + (height * np.cos(np.deg2rad(theta))) - (width * np.sin(np.deg2rad(theta))))

    print("rectangle found: ")
    print(x_1, y_1)
    print(x_2, y_2)
    print(x_3, y_3)
    print(x_4, y_4)
    print("=================")

    cv2.line(image, (x_1, y_1), (x_2, y_2), (255, 0, 255), 2)
    cv2.line(image, (x_2, y_2), (x_3, y_3), (255, 0, 255), 2)
    cv2.line(image, (x_3, y_3), (x_4, y_4), (255, 0, 255), 2)
    cv2.line(image, (x_4, y_4), (x_1, y_1), (255, 0, 255), 2)
