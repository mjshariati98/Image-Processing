import cv2
import numpy as np

from HW4.Exercise_3.helper import get_random_patch, get_most_similar_patch

sample = cv2.imread("../resources/sample.jpg")

width, height = 1010, 1010
image = np.zeros((height, width, 3))

PATCH_SIZE = 50
OVERLAP_SIZE = 10
STEP = PATCH_SIZE - OVERLAP_SIZE

for i in range(0, height - STEP, STEP):
    for j in range(0, width - STEP, STEP):
        if i == 0 and j == 0:  # left-top patch
            first_patch = get_random_patch(sample, PATCH_SIZE)
            image[:PATCH_SIZE, :PATCH_SIZE] = first_patch
        elif i == 0:  # first row
            previous_patch = image[:PATCH_SIZE, j - STEP: j + OVERLAP_SIZE]
            similar_patch = get_most_similar_patch(previous_patch=previous_patch,
                                                   sample=sample,
                                                   patch_size=PATCH_SIZE,
                                                   overlap_size=OVERLAP_SIZE,
                                                   frc='first_row')

            image[: PATCH_SIZE, j:j + PATCH_SIZE] = similar_patch

        elif j == 0:  # first column
            previous_patch = image[i - STEP:i + OVERLAP_SIZE, :PATCH_SIZE]
            similar_patch = get_most_similar_patch(previous_patch=previous_patch,
                                                   sample=sample,
                                                   patch_size=PATCH_SIZE,
                                                   overlap_size=OVERLAP_SIZE,
                                                   frc='first_column')

            image[i:i + PATCH_SIZE, : PATCH_SIZE] = similar_patch

        else:
            previous_patch = image[i:i + PATCH_SIZE, j: j + PATCH_SIZE]
            similar_patch = get_most_similar_patch(previous_patch=previous_patch,
                                                   sample=sample,
                                                   patch_size=PATCH_SIZE,
                                                   overlap_size=OVERLAP_SIZE,
                                                   frc='else')

            image[i:i + PATCH_SIZE, j: j + PATCH_SIZE] = similar_patch

cv2.imwrite("out/im3.jpg", image)
