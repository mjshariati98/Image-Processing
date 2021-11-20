import cv2
import numpy as np
from scipy import signal
import math
from HW2.Exercise_1.helper import gaussian_derivative_filter, row_gaussian_derivative_in_x_axis_filter, \
    row_gaussian_derivative_in_y_axis_filter, col_gaussian_derivative_in_x_axis_filter, \
    col_gaussian_derivative_in_y_axis_filter, get_distributed_image, check_equality, clean

# read image
source_image = cv2.imread(filename="../resources/books.jpg")

# convert to gray
gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

# filters
print("gaussian derivative filter in x-axis: ")
print(gaussian_derivative_filter(7, 1, 'x'))
print("==============================================")
print("gaussian derivative filter in y-axis: ")
print(gaussian_derivative_filter(7, 1, 'y'))

print("==============================================")

# row and column filters
print("Row gaussian derivative filter in x-axis: ")
print(row_gaussian_derivative_in_x_axis_filter(7, 1))
print("==============================================")
print("Col gaussian derivative filter in x-axis: ")
print(col_gaussian_derivative_in_x_axis_filter(7, 1))
print("==============================================")
print("Row gaussian derivative filter in y-axis: ")
print(row_gaussian_derivative_in_y_axis_filter(7, 1))
print("==============================================")
print("Col gaussian derivative filter in y-axis: ")
print(col_gaussian_derivative_in_y_axis_filter(7, 1))

print("==============================================")

# convolve main image with row filter in x-axis
hor_row = signal.convolve2d(gray_image, row_gaussian_derivative_in_x_axis_filter(7, 1), 'same')
cv2.imwrite("out/Q1_01_hor_row.jpg", hor_row)

# convolve main image with row filter in y-axis
ver_row = signal.convolve2d(gray_image, row_gaussian_derivative_in_y_axis_filter(7, 1), 'same')
cv2.imwrite("out/Q1_02_ver_row.jpg", ver_row)

# convolve hor_row image with col filter in x-axis
hor_col = signal.convolve2d(hor_row, col_gaussian_derivative_in_x_axis_filter(7, 1), 'same')
cv2.imwrite("out/Q1_03_hor-col.jpg", hor_col)

# convolve ver_row image with col filter in y-axis
ver_col = signal.convolve2d(ver_row, col_gaussian_derivative_in_y_axis_filter(7, 1), 'same')
cv2.imwrite("out/Q1_04_ver_col.jpg", ver_col)

# convolve 2D main image with 2d gaussian derivative filter in x-axis
x_2d = signal.convolve2d(gray_image, gaussian_derivative_filter(7, 1, 'x'), 'same')
cv2.imwrite("out/Q1_05_hor.jpg", x_2d)

# convolve 2D main image with 2d gaussian derivative filter in y-axis
y_2d = signal.convolve2d(gray_image, gaussian_derivative_filter(7, 1, 'y'), 'same')
cv2.imwrite("out/Q1_06_ver.jpg", y_2d)

print("x is equal to hor_col" if check_equality(x_2d, hor_col) else "x is different with hor_col")
print("y is equal to ver_col" if check_equality(y_2d, ver_col) else "y is different with ver_col")

print("==============================================")

# calculate gradian image
grad = np.hypot(x_2d, y_2d)
cv2.imwrite("out/Q1_07_grad_mag.jpg", grad)

# calculate theta image and save distribute version (0-255) of theta
theta = np.arctan2(y_2d, x_2d)
distributed_theta = get_distributed_image(theta)
cv2.imwrite("out/Q1_08_grad_dir.jpg", distributed_theta)

# clean image for threshold
clean_image = clean(grad, theta)

# distribute (0-255) clean image and remove pixels with less than 20 intensity (threshold)
distributed_clean_image = get_distributed_image(clean_image)
threshold_image = np.vectorize(lambda x: 255 if x > 20 else 0)(distributed_clean_image)
cv2.imwrite("out/Q1_09_edge.jpg", threshold_image)
