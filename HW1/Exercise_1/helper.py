import math
import numpy as np
import cv2


def show_image(image, resize_ratio):
    resized_image = cv2.resize(src=image, dsize=None, fx=resize_ratio, fy=resize_ratio)
    cv2.imshow(winname="image", mat=resized_image)
    cv2.waitKey(0)


def gamma_function(source_image, alpha):
    return pow(source_image / 255.0, alpha) * 255.0


def logarithm_function(source_image, alpha):
    return np.vectorize(lambda x: (255 * math.log(1 + alpha * x)) / math.log(1 + 255 * alpha))(source_image)
