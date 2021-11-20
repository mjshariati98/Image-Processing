import cv2
import numpy as np

from HW4.Exercise_1.helper import get_random_patch

sample = cv2.imread("../resources/sample.jpg")

width, height = 1000, 1000
image = np.zeros((height, width, 3))

PATCH_SIZE = 50

for i in range(height // PATCH_SIZE):
    for j in range(width // PATCH_SIZE):
        patch = get_random_patch(sample, PATCH_SIZE)
        image[i * PATCH_SIZE:i * PATCH_SIZE + PATCH_SIZE, j * PATCH_SIZE:j * PATCH_SIZE + PATCH_SIZE] = patch

cv2.imwrite("out/im1.jpg", image)
