import cv2
import numpy as np
import matplotlib.pyplot as plt

from HW2.Exercise_2.helper import get_edge_points, k_means, draw_rectangle, fill_a_b_theta_table, \
    get_rectangle_width_height_theta

NUMBER_OF_BOOKS = 3

# load image and edge image (get from Exercise1)
main_image = cv2.imread("../resources/Books.jpg")
edge_image = cv2.imread("Q1_09_edge.jpg", cv2.IMREAD_GRAYSCALE)
edge_image = np.vectorize(lambda x: 255 if x > 20 else 0)(edge_image)

# get points that are 255 in edge image for using in K-Means clustering
points = get_edge_points(edge_image)

# using k-means for divide points to 3(NUMBER_OF_BOOKS) clusters
cluster_centers, clusters = k_means(points, NUMBER_OF_BOOKS)
print("Cluster Cent‍ers: ")
print(cluster_centers)
print("Total Points: ", len(points))
print("number of points in each cluster: ")
print("cluster1 points: ", len(clusters[0]))
print("cluster2 points: ", len(clusters[1]))
print("cluster3 points: ", len(clusters[2]))

# show clusters
for cluster in range(NUMBER_OF_BOOKS):
    plt.scatter(clusters[cluster][:, 0], clusters[cluster][:, 1])
    plt.imshow(edge_image)
    plt.show()

# Book 1
center_x = 490
center_y = 250

A = 148  # 150
AA = 152
B = 94  # 96
BB = 98
table = np.zeros((AA - A, BB - B, 18))
table = fill_a_b_theta_table(clusters[0], table, center_x, center_y, A, AA, B, BB)
max = table.max()

rectangle_width, rectangle_height, rectangle_theta = get_rectangle_width_height_theta(table, A, AA, B, BB, max)

draw_rectangle(center_x, center_y, rectangle_width, rectangle_height, rectangle_theta, main_image)

# Book 2
center_x = 620
center_y = 886

A = 178  # 180
AA = 182
B = 118  # 120
BB = 122
table = np.zeros((AA - A, BB - B, 18))
table = fill_a_b_theta_table(clusters[1], table, center_x, center_y, A, AA, B, BB)

max = table.max()
rectangle_width, rectangle_height, rectangle_theta = get_rectangle_width_height_theta(table, A, AA, B, BB, max)

draw_rectangle(center_x, center_y, rectangle_width, rectangle_height, rectangle_theta, main_image)

# Book 3
center_x = 280
center_y = 584

A = 141  # 143
AA = 145
B = 98  # 100
BB = 102
table = np.zeros((AA - A, BB - B, 18))
table = fill_a_b_theta_table(clusters[2], table, center_x, center_y, A, AA, B, BB)

max = table.max()

rectangle_width, rectangle_height, rectangle_theta = get_rectangle_width_height_theta(table, A, AA, B, BB, max)

draw_rectangle(center_x, center_y, rectangle_width, rectangle_height, rectangle_theta, main_image)

cv2.imwrite("out/Q2.jpg", main_image)
