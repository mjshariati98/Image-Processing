from time import time
from skimage.segmentation import slic
import numpy as np
import math


def slic_oversegmentation(sharpen_image):
    t1 = time()
    segments = slic(sharpen_image, n_segments=10000, convert2lab=True, sigma=3)
    t2 = time()
    print(t2 - t1, " sec for over-segmentation")

    return segments


def get_around_pixels(x, y, radius):
    number_of_around_pixels = (radius ** 2) + ((radius - 1) ** 2)
    around_pixels = np.zeros((number_of_around_pixels, 2))

    index = 0
    for r in range(radius):
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if math.fabs(i) + math.fabs(j) == r:
                    around_pixels[index] = [x + i, y + j]
                    index += 1

    return around_pixels


def merge_super_pixels(birds, segments, segmented_image, radius, threshold):
    t1 = time()
    for i in range(birds.shape[0]):
        clusters = []
        birds_center = birds[i]
        birds_center_x = birds_center[0]
        birds_center_y = birds_center[1]
        if 0 < birds_center_x < segmented_image.shape[1] and 0 < birds_center_y < segmented_image.shape[0]:
            B, G, R = segmented_image[birds_center_y, birds_center_x]
            around_pixels = get_around_pixels(birds_center_x, birds_center_y, radius=radius)
            for x, y in around_pixels:
                x = int(x)
                y = int(y)

                if x < segmented_image.shape[1] and y < segmented_image.shape[0]:
                    neighbour_cluster = segments[y, x]
                    if neighbour_cluster in clusters:
                        continue
                    else:
                        clusters.append(neighbour_cluster)

                    if segments[y, x] != segments[birds_center_y, birds_center_x]:
                        b, g, r = segmented_image[y, x]
                        if int((b - B) ** 2 + (g - G) ** 2 + (r - R) ** 2) < threshold:
                            segments = np.vectorize(
                                lambda elem: segments[birds_center_y, birds_center_x] if elem == segments[
                                    y, x] else elem)(
                                segments)
    t2 = time()
    print((t2 - t1)/60, " min for merge super pixels")

    return segments
