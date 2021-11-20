import cv2
import numpy as np

from HW5.Exercise_1.helper import build_A_matrix, change_A, solve_equation, get_A_dot_s_image

for i in range(1, 3):
    # read source and target images and mask
    source_image = cv2.imread("../resources/q1-source" + str(i) + ".jpg")
    target_image = cv2.imread("../resources/q1-target" + str(i) + ".jpg")
    mask_image = cv2.cvtColor(cv2.imread("../resources/q1-mask" + str(i) + ".png"), cv2.COLOR_BGR2GRAY)

    # get height and width of images
    height, width = source_image.shape[:2]

    # change 255 values on mask to 1
    mask = np.vectorize(lambda x: 1 if x is 255 else 0)(mask_image)

    # get A matrix (use to solve poisson equation) and
    A = build_A_matrix(height, width)

    # get A * s result to see act like gradient (As.jpg in out folder)
    As = get_A_dot_s_image(A, source_image, height, width)
    cv2.imwrite("out/As" + str(i) + ".jpg", As)

    # change A to get exactly target pixels outside the mask region
    A = change_A(A, mask)

    # separate source image to its channels
    source_B_channel = source_image[:, :, 0]
    source_G_channel = source_image[:, :, 1]
    source_R_channel = source_image[:, :, 2]

    # separate target image to its channels
    target_B_channel = target_image[:, :, 0]
    target_G_channel = target_image[:, :, 1]
    target_R_channel = target_image[:, :, 2]

    # solve equation Af=b for each channel
    f_B = solve_equation(source_B_channel, target_B_channel, mask, A)
    f_G = solve_equation(source_G_channel, target_G_channel, mask, A)
    f_R = solve_equation(source_R_channel, target_R_channel, mask, A)

    # get result image from f_B, f_G and f_R(from solving above equations)
    result_image = np.zeros(source_image.shape)
    result_image[:, :, 0] = f_B
    result_image[:, :, 1] = f_G
    result_image[:, :, 2] = f_R

    # save result
    cv2.imwrite("out/im" + str(i) + ".jpg", result_image)
