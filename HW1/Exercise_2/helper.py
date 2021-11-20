import numpy as np
import binary_search
import cv2
from matplotlib import pyplot as plt


def show_image(image, resize_ratio):
    resized_image = cv2.resize(src=image, dsize=None, fx=resize_ratio, fy=resize_ratio)
    cv2.imshow(winname="image", mat=resized_image)
    cv2.waitKey(0)


def show_histogram(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


def calculate_cdf_from_histogram(hist):
    cdf = np.zeros(hist.shape[0])
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
    cdf = np.vectorize(lambda x: x / cdf[-1])(cdf)
    return cdf


def find_map_between_source_and_target_cdf(source_cdf, target_cdf):
    map = []
    for i in range(0, 256):
        closest_index = binary_search.search(target_cdf, source_cdf[i])
        map.append(closest_index)

    return map
