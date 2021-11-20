import random
from builtins import min

import cv2
import numpy as np


def get_random_patch(sample, patch_size):
    height, width = sample.shape[0:2]
    i = random.randrange(0, height - patch_size)
    j = random.randrange(0, width - patch_size)

    return sample[i:i + patch_size, j:j + patch_size]


def get_most_similar_patch(previous_patch, sample, patch_size, overlap_size, frc):
    """
    give the most similar patch from sample to out patch. use SSD method.
    :param previous_patch: the patch that want to find the most similar patch to it.
    :param sample: out sample texture.
    :param patch_size: the size of patch.
    :param overlap_size: the size that two patch will overlap.
    :param frc: first row or column or another parts. a boolean that specify whether this 'template matching' belongs to
     completing first row or column or other parts.
    :return: most similar patch to given patch(previous_patch)
    """
    step = patch_size - overlap_size
    height, width = sample.shape[:2]

    previous_patch_gray = cv2.cvtColor(np.array(previous_patch, dtype='uint8'), cv2.COLOR_BGR2GRAY)
    sample_gray = cv2.cvtColor(np.array(sample, dtype='uint8'), cv2.COLOR_BGR2GRAY)

    if frc == 'first_row':  # first row or first column
        prev_overlap_patch = previous_patch_gray[:, step:patch_size]
        minimum = 99999999

        dict = {}
        sample_gray = np.array(sample_gray, dtype='float64')
        prev_overlap_patch = np.array(prev_overlap_patch, dtype='float64')
        for i in range(0, height - patch_size + 1):
            for j in range(0, width - patch_size + 1):

                diff = sample_gray[i:i + patch_size, j:j + overlap_size] - prev_overlap_patch
                diff = np.power(diff, 2)
                sum = np.sum(diff)
                if sum < minimum:
                    minimum = sum

                if dict.keys().__contains__(sum):
                    dict[sum].append((i, j))
                else:
                    dict[sum] = [(i, j)]

        choices = []
        for s in dict.keys():
            if (minimum != 0 and s < 1.2 * minimum) or (s < 100):
                for elem in dict[s]:
                    choices.append(elem)

        r = random.randrange(len(choices))
        best_i, best_j = choices[r]

        overlap = sample_gray[best_i:best_i + patch_size, best_j:best_j + overlap_size] - prev_overlap_patch
        overlap = np.power(overlap, 2)
        h, w = overlap.shape[:2]

        path = get_best_path(overlap)

        result_overlap_patch = np.zeros((patch_size, overlap_size, 3))
        for i in range(h):
            for j in range(w):
                if j <= path[h - i - 1]:
                    result_overlap_patch[i, j] = previous_patch[i, step + j]
                else:
                    result_overlap_patch[i, j] = sample[best_i + i, best_j + j]

        result_patch = np.zeros((patch_size, patch_size, 3))
        result_patch[:, :overlap_size] = result_overlap_patch
        result_patch[:, overlap_size:patch_size] = sample[best_i:best_i + patch_size,
                                                   best_j + overlap_size:best_j + patch_size]

        return result_patch

    elif frc == 'first_column':
        prev_overlap_patch = previous_patch_gray[step:patch_size, :]
        minimum = 99999999
        dict = {}

        sample_gray = np.array(sample_gray, dtype='float64')
        prev_overlap_patch = np.array(prev_overlap_patch, dtype='float64')
        for i in range(0, height - patch_size + 1):
            for j in range(0, width - patch_size + 1):
                diff = sample_gray[i:i + overlap_size, j:j + patch_size] - prev_overlap_patch
                diff = np.power(diff, 2)
                sum = np.sum(diff)

                if sum < minimum:
                    minimum = sum

                if dict.keys().__contains__(sum):
                    dict[sum].append((i, j))
                else:
                    dict[sum] = [(i, j)]

        choices = []
        for s in dict.keys():
            if (minimum != 0 and s < 1.2 * minimum) or (s < 100):
                for elem in dict[s]:
                    choices.append(elem)

        r = random.randrange(len(choices))
        best_i, best_j = choices[r]

        overlap = sample_gray[best_i:best_i + overlap_size, best_j:best_j + patch_size] - prev_overlap_patch
        overlap = np.power(overlap, 2)
        h, w = overlap.shape[:2]

        path = get_best_path(overlap.T)
        result_overlap_patch = np.zeros((overlap_size, patch_size, 3))

        for i in range(w):
            for j in range(h):
                if j <= path[w - i - 1]:
                    result_overlap_patch[j, i] = previous_patch[step + j, i]
                else:
                    result_overlap_patch[j, i] = sample[best_i + j, best_j + i]

        result_patch = np.zeros((patch_size, patch_size, 3))
        result_patch[:overlap_size, :] = result_overlap_patch
        result_patch[overlap_size:, :patch_size] = sample[best_i + overlap_size:best_i + patch_size,
                                                   best_j:best_j + patch_size]

        return result_patch

    else:
        minimum = 99999999

        dict = {}

        previous_patch_gray = np.array(previous_patch_gray, dtype='float64')
        sample_gray = np.array(sample_gray, dtype='float64')
        for i in range(0, height - patch_size + 1):
            for j in range(0, width - patch_size + 1):
                tmp_sample = np.array(sample_gray, dtype='float64')
                tmp_sample[i + overlap_size:i + patch_size, j + overlap_size:j + patch_size] = 0

                diff = tmp_sample[i:i + patch_size, j:j + patch_size] - previous_patch_gray
                diff = np.power(diff, 2)
                sum = np.sum(diff)
                if sum < minimum:
                    minimum = sum

                if dict.keys().__contains__(sum):
                    dict[sum].append((i, j))
                else:
                    dict[sum] = [(i, j)]

        choices = []
        for s in dict.keys():
            if (minimum != 0 and s < 1.2 * minimum) or (s < 50):
                for elem in dict[s]:
                    choices.append(elem)
        r = random.randrange(len(choices))
        best_i, best_j = choices[r]


        result_patch = np.zeros((patch_size, patch_size, 3))

        # left
        prev_overlap_patch = previous_patch_gray[:, step:patch_size]
        overlap = sample_gray[best_i:best_i + patch_size, best_j:best_j + overlap_size] - prev_overlap_patch
        overlap = np.power(overlap, 2)
        h, w = overlap.shape[:2]

        path = get_best_path(overlap)

        result_overlap_patch = np.zeros((patch_size, overlap_size, 3))
        for i in range(h):
            for j in range(w):
                if j <= path[h - i - 1]:  # prev
                    result_overlap_patch[i, j] = previous_patch[i, j]
                else:
                    result_overlap_patch[i, j] = sample[best_i + i, best_j + j]

        result_patch[:, :overlap_size] = result_overlap_patch

        # top
        prev_overlap_patch = previous_patch_gray[step:patch_size, :]
        overlap = sample_gray[best_i:best_i + overlap_size, best_j:best_j + patch_size] - prev_overlap_patch
        overlap = np.power(overlap, 2)
        h, w = overlap.shape[:2]

        path = get_best_path(overlap.T)
        result_overlap_patch = np.zeros((overlap_size, patch_size, 3))

        for i in range(w):
            for j in range(h):
                if j <= path[w - i - 1]:
                    result_overlap_patch[j, i] = previous_patch[j, i]
                else:
                    result_overlap_patch[j, i] = sample[best_i + j, best_j + i]

        result_patch[:overlap_size, :] = result_overlap_patch

        result_patch[overlap_size:, overlap_size:] = sample[best_i + overlap_size:best_i + patch_size,
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
