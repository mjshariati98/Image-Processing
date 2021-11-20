import random
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
        min = 99999999
        best_i = 0
        best_j = 0
        for i in range(0, height - patch_size + 1):
            for j in range(0, width - patch_size + 1):
                diff = sample_gray[i:i + patch_size, j:j + overlap_size] - prev_overlap_patch
                diff = np.power(diff, 2)
                sum = np.sum(diff)
                if sum <= min:
                    min = sum
                    best_i = i
                    best_j = j

        return sample[best_i:best_i + patch_size, best_j:best_j + patch_size]
    elif frc == 'first_column':
        prev_overlap_patch = previous_patch_gray[step:patch_size, :]
        min = 99999999
        best_i = 0
        best_j = 0
        for i in range(0, height - patch_size + 1):
            for j in range(0, width - patch_size + 1):
                diff = sample_gray[i:i + overlap_size, j:j + patch_size] - prev_overlap_patch
                diff = np.power(diff, 2)
                sum = np.sum(diff)

                if sum < min:
                    min = sum
                    best_i = i
                    best_j = j

        return sample[best_i:best_i + patch_size, best_j:best_j + patch_size]

    else:
        min = 99999999
        best_i = 0
        best_j = 0
        for i in range(0, height - patch_size + 1):
            for j in range(0, width - patch_size + 1):
                previous_patch_gray = np.array(previous_patch_gray, dtype='float64')
                tmp_sample = np.array(sample_gray, dtype='float64')
                tmp_sample[i + overlap_size:i + patch_size, j + overlap_size:j + patch_size] = 0

                diff = tmp_sample[i:i + patch_size, j:j + patch_size] - previous_patch_gray
                diff = np.power(diff, 2)
                sum = np.sum(diff)
                if sum < min:
                    min = sum
                    best_i = i
                    best_j = j

        return sample[best_i:best_i + patch_size, best_j:best_j + patch_size]
