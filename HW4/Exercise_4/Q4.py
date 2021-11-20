import cv2
import numpy as np

from HW4.Exercise_4.helper import get_first_patch, get_most_similar_patch

texture = cv2.imread("../resources/texture.png")
picture = cv2.imread("../resources/picture.png")

height, width = picture.shape[:2]
image = np.zeros((height, width, 3))

PATCH_SIZE = 20
OVERLAP_SIZE = 4
STEP = PATCH_SIZE - OVERLAP_SIZE
ALPHA = 0.8

for i in range(0, height - STEP, STEP):
    print(i)
    for j in range(0, width - STEP, STEP):
        if i == 0 and j == 0:  # left-top patch
            picture_piece = picture[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            first_patch = get_first_patch(texture, picture_piece, PATCH_SIZE)
            image[0:PATCH_SIZE, 0:PATCH_SIZE] = first_patch
        elif i == 0:  # first row
            previous_patch = image[:PATCH_SIZE, j - STEP: j + OVERLAP_SIZE]
            picture_piece = picture[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            similar_patch = get_most_similar_patch(previous_patch=previous_patch,
                                                   texture=texture,
                                                   picture=picture_piece,
                                                   patch_size=PATCH_SIZE,
                                                   overlap_size=OVERLAP_SIZE,
                                                   alpha=ALPHA,
                                                   frc='first_row')

            image[: PATCH_SIZE, j:j + PATCH_SIZE] = similar_patch

        elif j == 0:  # first column
            previous_patch = image[i - STEP:i + OVERLAP_SIZE, :PATCH_SIZE]
            picture_piece = picture[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            similar_patch = get_most_similar_patch(previous_patch=previous_patch,
                                                   texture=texture,
                                                   picture=picture_piece,
                                                   patch_size=PATCH_SIZE,
                                                   overlap_size=OVERLAP_SIZE,
                                                   alpha=ALPHA,
                                                   frc='first_column')

            image[i:i + PATCH_SIZE, : PATCH_SIZE] = similar_patch

        else:
            previous_patch = image[i:i + PATCH_SIZE, j: j + PATCH_SIZE]
            picture_piece = picture[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            similar_patch = get_most_similar_patch(previous_patch=previous_patch,
                                                   texture=texture,
                                                   picture=picture_piece,
                                                   patch_size=PATCH_SIZE,
                                                   overlap_size=OVERLAP_SIZE,
                                                   alpha=ALPHA,
                                                   frc='else')

            image[i:i + PATCH_SIZE, j: j + PATCH_SIZE] = similar_patch

cv2.imwrite("out/im4.jpg", image)
