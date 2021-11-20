import cv2
import numpy as np

from HW4.Exercise_2.helper import get_random_patch, get_most_similar_patch

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
            image[0:PATCH_SIZE, 0:PATCH_SIZE] = first_patch
        elif i == 0:  # first row
            previous_patch = image[:PATCH_SIZE, j - STEP: j + OVERLAP_SIZE]
            similar_patch = get_most_similar_patch(previous_patch=previous_patch,
                                                   sample=sample,
                                                   patch_size=PATCH_SIZE,
                                                   overlap_size=OVERLAP_SIZE,
                                                   frc='first_row')

            previous_overlap_patch = previous_patch[:, STEP:PATCH_SIZE]
            similar_overlap_patch = similar_patch[:, STEP:PATCH_SIZE]
            avg_overlap_patch = (previous_overlap_patch + similar_overlap_patch) / 2

            image[:PATCH_SIZE, j:j + OVERLAP_SIZE] = avg_overlap_patch
            image[: PATCH_SIZE, j + OVERLAP_SIZE:j + PATCH_SIZE] = similar_patch[:, OVERLAP_SIZE:PATCH_SIZE]

        elif j == 0:  # first column
            previous_patch = image[i - STEP:i + OVERLAP_SIZE, :PATCH_SIZE]
            similar_patch = get_most_similar_patch(previous_patch=previous_patch,
                                                   sample=sample,
                                                   patch_size=PATCH_SIZE,
                                                   overlap_size=OVERLAP_SIZE,
                                                   frc='first_column')

            previous_overlap_patch = previous_patch[STEP:PATCH_SIZE, :]
            similar_overlap_patch = similar_patch[:OVERLAP_SIZE, :]
            avg_overlap_patch = (previous_overlap_patch + similar_overlap_patch) / 2

            image[i:i + OVERLAP_SIZE, :PATCH_SIZE] = avg_overlap_patch
            image[i + OVERLAP_SIZE:i + PATCH_SIZE, : PATCH_SIZE] = similar_patch[OVERLAP_SIZE:PATCH_SIZE, :]

        else:
            previous_patch = image[i:i + PATCH_SIZE, j: j + PATCH_SIZE]
            similar_patch = get_most_similar_patch(previous_patch=previous_patch,
                                                   sample=sample,
                                                   patch_size=PATCH_SIZE,
                                                   overlap_size=OVERLAP_SIZE,
                                                   frc='else')

            left_previous_overlap_patch = previous_patch[:, :OVERLAP_SIZE]
            top_previous_overlap_patch = previous_patch[:OVERLAP_SIZE, :]
            left_similar_overlap_patch = similar_patch[:, :OVERLAP_SIZE]
            top_similar_overlap_patch = similar_patch[:OVERLAP_SIZE, :]
            left_avg_overlap_patch = (left_previous_overlap_patch + left_similar_overlap_patch) / 2
            top_avg_overlap_patch = (top_previous_overlap_patch + top_similar_overlap_patch) / 2

            image[i:i + PATCH_SIZE, j:j + OVERLAP_SIZE] = left_avg_overlap_patch
            image[i:i + OVERLAP_SIZE, j:j + PATCH_SIZE] = top_avg_overlap_patch
            image[i + OVERLAP_SIZE:i + PATCH_SIZE, j + OVERLAP_SIZE: j + PATCH_SIZE] = \
                similar_patch[OVERLAP_SIZE:PATCH_SIZE, OVERLAP_SIZE:PATCH_SIZE]


cv2.imwrite("out/im2.jpg", image)
