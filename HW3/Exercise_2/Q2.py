from time import time
import cv2
from sklearn.cluster import MeanShift

from HW3.Exercise_2.helper import build_RGBXY_array, show_clusters

source_image = cv2.imread("../resources/IMG_2805.JPG")
img = cv2.resize(src=source_image, dsize=None, fx=1 / 8, fy=1 / 8)
width = img.shape[1]
height = img.shape[0]

points = build_RGBXY_array(img, height, width)

t1 = time()
mean_shift = MeanShift(bandwidth=40, n_jobs=-1).fit(points)
t2 = time()
print((t2 - t1) / 60, " min")

show_clusters(points, mean_shift.labels_, width, height)

# with 8 cores -> 38.00329860448837 min
