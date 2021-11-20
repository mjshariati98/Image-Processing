import random
import cv2
import numpy as np


def get_first_patch(texture, picture, patch_size):
    height, width = texture.shape[:2]
    picture_gray = cv2.cvtColor(np.array(picture, dtype='uint8'), cv2.COLOR_BGR2GRAY)
    texture_gray = cv2.cvtColor(np.array(texture, dtype='uint8'), cv2.COLOR_BGR2GRAY)

    min = 99999999
    best_i = 0
    best_j = 0

    for i in range(0, height - patch_size + 1):
        for j in range(0, width - patch_size + 1):
            picture_gray = np.array(picture_gray, dtype='float64')
            texture_gray = np.array(texture_gray, dtype='float64')

            diff_picture = texture_gray[i:i + patch_size, j:j + patch_size] - picture_gray
            diff_picture = np.power(diff_picture, 2)
            cost_picture = np.sum(diff_picture)

            if cost_picture < min:
                min = cost_picture
                best_i = i
                best_j = j

    return texture[best_i:best_i + patch_size, best_j:best_j + patch_size]


def get_most_similar_patch(previous_patch, texture, picture, patch_size, overlap_size, alpha, frc):
    """
    give the most similar patch from sample to out patch. use SSD method.
    :param previous_patch: the patch that want to find the most similar patch to it.
    :param texture: out sample texture.
    :param patch_size: the size of patch.
    :param overlap_size: the size that two patch will overlap.
    :param frc: first row or column or another parts. a boolean that specify whether this 'template matching' belongs to
     completing first row or column or other parts.
    :return: most similar patch to given patch(previous_patch)
    """
    step = patch_size - overlap_size
    height, width = texture.shape[:2]

    previous_patch_gray = cv2.cvtColor(np.array(previous_patch, dtype='uint8'), cv2.COLOR_BGR2GRAY)
    texture_gray = cv2.cvtColor(np.array(texture, dtype='uint8'), cv2.COLOR_BGR2GRAY)
    picture_gray = cv2.cvtColor(np.array(picture, dtype='uint8'), cv2.COLOR_BGR2GRAY)

    if frc == 'first_row':  # first row or first column
        prev_overlap_patch = previous_patch_gray[:, step:patch_size]
        minimum = 99999999
        best_i = 0
        best_j = 0
        for i in range(0, height - patch_size + 1):
            for j in range(0, width - patch_size + 1):
                diff_texture = texture_gray[i:i + patch_size, j:j + overlap_size] - prev_overlap_patch
                diff_texture = np.power(diff_texture, 2)
                cost_texture = np.sum(diff_texture)

                picture_gray = np.array(picture_gray, dtype='float64')
                texture_gray = np.array(texture_gray, dtype='float64')

                diff_picture = texture_gray[i:i + patch_size, j:j + patch_size] - picture_gray
                diff_picture = np.power(diff_picture, 2)
                cost_picture = np.sum(diff_picture)

                cost = (alpha * cost_texture) + ((1 - alpha) * cost_picture)

                if cost <= minimum:
                    minimum = cost
                    best_i = i
                    best_j = j

        overlap = texture_gray[best_i:best_i + patch_size, best_j:best_j + overlap_size] - prev_overlap_patch
        overlap = np.power(overlap, 2)
        h, w = overlap.shape[:2]

        path = get_best_path(overlap)

        result_overlap_patch = np.zeros((patch_size, overlap_size, 3))
        for i in range(h):
            for j in range(w):
                if j <= path[h - i - 1]:
                    result_overlap_patch[i, j] = previous_patch[i, step + j]
                else:
                    result_overlap_patch[i, j] = texture[best_i + i, best_j + j]

        result_patch = np.zeros((patch_size, patch_size, 3))
        result_patch[:, :overlap_size] = result_overlap_patch
        result_patch[:, overlap_size:patch_size] = texture[best_i:best_i + patch_size,
                                                   best_j + overlap_size:best_j + patch_size]

        return result_patch

    elif frc == 'first_column':
        prev_overlap_patch = previous_patch_gray[step:patch_size, :]
        minimum = 99999999
        best_i = 0
        best_j = 0
        for i in range(0, height - patch_size + 1):
            for j in range(0, width - patch_size + 1):
                picture_gray = np.array(picture_gray, dtype='float64')
                texture_gray = np.array(texture_gray, dtype='float64')

                diff_texture = texture_gray[i:i + overlap_size, j:j + patch_size] - prev_overlap_patch
                diff_texture = np.power(diff_texture, 2)
                cost_texture = np.sum(diff_texture)

                diff_picture = texture_gray[i:i + patch_size, j:j + patch_size] - picture_gray
                diff_picture = np.power(diff_picture, 2)
                cost_picture = np.sum(diff_picture)

                cost = (alpha * cost_texture) + ((1 - alpha) * cost_picture)

                if cost < minimum:
                    minimum = cost
                    best_i = i
                    best_j = j

        overlap = texture_gray[best_i:best_i + overlap_size, best_j:best_j + patch_size] - prev_overlap_patch
        overlap = np.power(overlap, 2)
        h, w = overlap.shape[:2]

        path = get_best_path(overlap.T)
        result_overlap_patch = np.zeros((overlap_size, patch_size, 3))

        for i in range(w):
            for j in range(h):
                if j <= path[w - i - 1]:
                    result_overlap_patch[j, i] = previous_patch[step + j, i]
                else:
                    result_overlap_patch[j, i] = texture[best_i + j, best_j + i]

        result_patch = np.zeros((patch_size, patch_size, 3))
        result_patch[:overlap_size, :] = result_overlap_patch
        result_patch[overlap_size:, :patch_size] = texture[best_i + overlap_size:best_i + patch_size,
                                                   best_j:best_j + patch_size]

        return result_patch

    else:
        minimum = 99999999
        best_i = 0
        best_j = 0

        picture_gray = np.array(picture_gray, dtype='float64')
        texture_gray = np.array(texture_gray, dtype='float64')
        previous_patch_gray = np.array(previous_patch_gray, dtype='float64')

        for i in range(0, height - patch_size + 1):
            for j in range(0, width - patch_size + 1):
                tmp_sample = np.array(texture_gray, dtype='float64')
                tmp_sample[i + overlap_size:i + patch_size, j + overlap_size:j + patch_size] = 0

                diff_texture = tmp_sample[i:i + patch_size, j:j + patch_size] - previous_patch_gray
                diff_texture = np.power(diff_texture, 2)
                cost_texture = np.sum(diff_texture)

                diff_picture = texture_gray[i:i + patch_size, j:j + patch_size] - picture_gray
                diff_picture = np.power(diff_picture, 2)
                cost_picture = np.sum(diff_picture)

                cost = (alpha * cost_texture) + ((1 - alpha) * cost_picture)

                if cost < minimum:
                    minimum = cost
                    best_i = i
                    best_j = j

        result_patch = np.zeros((patch_size, patch_size, 3))

        # left
        prev_overlap_patch = previous_patch_gray[:, step:patch_size]
        overlap = texture_gray[best_i:best_i + patch_size, best_j:best_j + overlap_size] - prev_overlap_patch
        overlap = np.power(overlap, 2)
        h, w = overlap.shape[:2]

        path = get_best_path(overlap)

        result_overlap_patch = np.zeros((patch_size, overlap_size, 3))
        for i in range(h):
            for j in range(w):
                if j <= path[h - i - 1]:  # prev
                    result_overlap_patch[i, j] = previous_patch[i, j]
                else:
                    result_overlap_patch[i, j] = texture[best_i + i, best_j + j]

        result_patch[:, :overlap_size] = result_overlap_patch

        # top
        prev_overlap_patch = previous_patch_gray[step:patch_size, :]
        overlap = texture_gray[best_i:best_i + overlap_size, best_j:best_j + patch_size] - prev_overlap_patch
        overlap = np.power(overlap, 2)
        h, w = overlap.shape[:2]

        path = get_best_path(overlap.T)
        result_overlap_patch = np.zeros((overlap_size, patch_size, 3))

        for i in range(w):
            for j in range(h):
                if j <= path[w - i - 1]:
                    result_overlap_patch[j, i] = previous_patch[j, i]
                else:
                    result_overlap_patch[j, i] = texture[best_i + j, best_j + i]

        result_patch[:overlap_size, :] = result_overlap_patch

        result_patch[overlap_size:, overlap_size:] = texture[best_i + overlap_size:best_i + patch_size,
                                                     best_j + overlap_size:best_j + patch_size]

        return result_patch


def get_best_path(overlap):
    E = np.zeros(overlap.shape, dtype='float64')
    h, w = overlap.shape[:2]

    for i in range(h):
        for j in range(w):
            if i == 0:
                E[i, j] = overlap[i, j]
            elif j == 0:
                E[i, j] = overlap[i, j] + min(E[i - 1, j], E[i - 1, j + 1])
            elif j == w - 1:
                E[i, j] = overlap[i, j] + min(E[i - 1, j - 1], E[i - 1, j])
            else:
                E[i, j] = overlap[i, j] + min(E[i - 1, j - 1], E[i - 1, j], E[i - 1, j + 1])

    path = []

    index = get_index(E, h - 1, -1)
    path.append(index)

    for i in range(h - 2, -1, -1):
        index = get_index(E, i, index)
        path.append(index)

    return path


def get_index(E, row, index):
    if index == -1:
        minimum = min(E[row])
        for i in range(len(E[row])):
            if E[row, i] == minimum:
                index = i
    else:
        if index == 0:
            minimum = min(E[row, index:index + 2])
            for i in range(index, index + 2):
                if E[row, i] == minimum:
                    index = i
        elif index == E.shape[1] - 1:
            minimum = min(E[row, index - 1:index + 1])
            for i in range(index - 1, index + 1):
                if E[row, i] == minimum:
                    index = i
        else:
            minimum = min(E[row, index - 1:index + 2])
            for i in range(index - 1, index + 2):
                if E[row, i] == minimum:
                    index = i

    return index
