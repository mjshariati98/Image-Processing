from time import time
import numpy as np
import math


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


def merge_super_pixels(image, birds, segments, radius, threshold):
    height, width = image.shape[0:2]
    t1 = time()

    for i in range(birds.shape[0]):
        clusters = []
        birds_center = birds[i]
        birds_center_x = birds_center[0]
        birds_center_y = birds_center[1]
        if 0 < birds_center_x < width and 0 < birds_center_y < height:
            B, G, R = image[birds_center_y, birds_center_x]
            around_pixels = get_around_pixels(birds_center_x, birds_center_y, radius=radius)
            for x, y in around_pixels:
                x = int(x)
                y = int(y)

                if x < width and y < height:
                    neighbour_cluster = segments[y, x]
                    if neighbour_cluster in clusters:
                        continue
                    else:
                        clusters.append(neighbour_cluster)

                    if segments[y, x] != segments[birds_center_y, birds_center_x]:
                        b, g, r = image[y, x]
                        if (int(b - B)) ** 2 + (int(g - G)) ** 2 + (int(r - R) ** 2) < threshold:
                            segments = np.vectorize(lambda elem: segments[birds_center_y, birds_center_x] if elem == segments[y, x] else elem)(segments)
    t2 = time()
    print((t2 - t1) / 60, " min for merge super pixels")

    return segments
