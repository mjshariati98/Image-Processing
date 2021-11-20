import cv2

from HW3.Exercise_1.helper import get_points, create_image_from_points, k_means, mean_shift, \
    get_points_in_polar_coordinates, create_image_from_polar_points, get_array_points

file = open("../resources/Points.txt", "r")

points_count, x, y, min_x, max_x, min_y, max_y = get_points(file)

width, height = max_x - min_x + 1, max_y - min_y + 1

image = create_image_from_points(x, y, min_x, max_x, min_y, max_y)
cv2.imwrite("out/im01.jpg", image)

points = get_array_points(x, y)

k_means(points,x, y, width, height, "im02")

mean_shift(x, y, width, height, "im03")

r, theta, min_r, max_r, min_theta, max_theta = get_points_in_polar_coordinates(file)

polar_image = create_image_from_polar_points(r, theta, max_r, min_theta, max_theta)
cv2.imwrite("out/im04-r-theta.jpg", polar_image)

polar_points = get_array_points(r, theta)
k_means(polar_points, x, y, width, height, "im04")
