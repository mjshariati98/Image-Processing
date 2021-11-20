import random


def get_random_patch(sample, patch_size):
    height, width = sample.shape[0:2]
    i = random.randrange(0, height - patch_size)
    j = random.randrange(0, width - patch_size)

    return sample[i:i + patch_size, j:j + patch_size]
