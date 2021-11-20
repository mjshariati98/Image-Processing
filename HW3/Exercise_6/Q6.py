import numpy as np
import cv2
import matplotlib.pyplot as plt

from HW3.Exercise_6.helper import normalize_image, get_dots, remove_out_of_circle, get_gradients, calculate_move, \
    make_video

file_path = "../resources/tasbih.jpg"
dots_count = 100
alpha, beta, gamma = 0.0001, 0.01, 0.01

source_image = cv2.imread(file_path)
height, width = source_image.shape[0], source_image.shape[1]

normalize_image = normalize_image(source_image)

image_gray = cv2.cvtColor(normalize_image, cv2.COLOR_BGR2GRAY)

dots, center_x, center_y, radius = get_dots(height, width, dots_count)

plt.imshow(source_image)
plt.scatter(dots[:, 0], dots[:, 1])
plt.show()

image = remove_out_of_circle(image_gray, center_x, center_y, radius)

gradient_x, gradient_y = get_gradients(image_gray)
cv2.imwrite("out/gradient_x.jpg", np.array(gradient_x, dtype='uint8'))
cv2.imwrite("out/gradient_y.jpg", np.array(gradient_y, dtype='uint8'))

for j in range(500):
    dots = calculate_move(dots, dots_count, height, width, alpha, beta, gamma, gradient_x, gradient_y)

    plt.imshow(source_image)
    plt.scatter(dots[:, 0], dots[:, 1], color='blue')
    # plt.show()
    plt.savefig("frames/" + str(j) + ".jpg")
    plt.clf()

size = (width, height)
make_video(size)
