import cv2
import numpy as np


def show_image(image, resize_ratio):
    resized_image = cv2.resize(src=image, dsize=None, fx=resize_ratio, fy=resize_ratio)
    cv2.imshow(winname="image", mat=resized_image)
    cv2.waitKey(0)


def get_height_and_width(image):
    height = image.shape[0]
    width = image.shape[1]

    return height, width


def get_BGR_channels(image):
    height = image.shape[0]

    B_channel = image[:height // 3, :, 0]
    G_channel = image[height // 3:2 * height // 3, :, 0]
    R_channel = image[2 * height // 3:, :, 0]

    return B_channel, G_channel, R_channel


def find_match_of_2_channels(mines_of_two_matrix, i, j, min, best_i, best_j, L):
    if L is 1:
        sum = np.sum(np.abs(mines_of_two_matrix))
    else:
        mines_of_two_matrix = np.power(mines_of_two_matrix, L)
        sum = np.sum(mines_of_two_matrix)

    if sum < min:
        min = sum
        best_i = i
        best_j = j

    return min, best_i, best_j


def find_match_B_and_G_channel(source_image, result_image, L):
    B_channel, G_channel, R_channel = get_BGR_channels(source_image)
    result_height, result_width = get_height_and_width(result_image)

    min = 999999999999999999999
    best_i = 0
    best_j = 0
    for i in range(35, 45):
        for j in range(15, 25):
            mat = np.zeros((B_channel.shape[0], B_channel.shape[1]))
            mat[i:, :result_width - j] = G_channel[:result_height - i, j:]
            mines_of_two_matrix = np.subtract(B_channel[i:, :result_width - j],
                                              mat[:result_height - i, j:])

            min, best_i, best_j = find_match_of_2_channels(mines_of_two_matrix, i, j, min, best_i, best_j, L)

    print(best_i)  # 43 #TODO
    print(best_j)  # 9
    return best_i, best_j


def fix_G_channel(result_image, G_channel, best_i, best_j):
    result_height, result_width = get_height_and_width(result_image)
    result_image[best_i:, :result_width - best_j, 1] = G_channel[:result_height - best_i, best_j:]

    return result_image


def find_match_B_and_R_channel(source_image, result_image, L):
    B_channel, G_channel, R_channel = get_BGR_channels(source_image)
    result_height, result_width = get_height_and_width(result_image)

    min = 999999999999999999999
    best_i = 0
    best_j = 0
    for i in range(100, 110):
        for j in range(0, 15):
            mat = np.zeros((B_channel.shape[0], B_channel.shape[1]))
            mat[i:, j:] = R_channel[:result_height - i, :result_width - j]
            mines_of_two_matrix = np.subtract(B_channel[i:, :result_width - j], mat[:result_height - i, j:])
            min, best_i, best_j = find_match_of_2_channels(mines_of_two_matrix, i, j, min, best_i, best_j, L)

    print(best_i)
    print(best_j)
    return best_i, best_j


def fix_R_channel(result_image, R_channel, best_i, best_j):
    result_height, result_width = get_height_and_width(result_image)
    result_image[best_i:, :result_width - best_j:, 2] = R_channel[:result_height - best_i, best_j:]

    return result_image
