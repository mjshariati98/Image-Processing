import cv2
import numpy as np
from HW1.Exercise_2.helper import calculate_cdf_from_histogram, find_map_between_source_and_target_cdf, show_image, \
    show_histogram

# read source image
source_image = cv2.imread(filename='../resources/IMG_2919.JPG')

# convert source image from RGB to HSV
source_hsv_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)

# # get V layer of source image
source_V_layer = source_hsv_image[:, :, 2]

# get histogram and calculate cdf from v layer of source image
source_hist = cv2.calcHist([source_hsv_image], [2], None, [256], [0, 256])
source_cdf = calculate_cdf_from_histogram(source_hist)
show_histogram(source_V_layer)

# we want target histogram be uniform, then we can write cdf like this: [0, 1, 2, ..., 255] / 255
target_cdf = np.vectorize(lambda x: x / 255)(np.arange(256))

# find map between source and target cdf
source_to_target_map = find_map_between_source_and_target_cdf(source_cdf, target_cdf)

flat_V_layer = source_V_layer.ravel()
for i in range(flat_V_layer.size):
    intensity = flat_V_layer[i]
    flat_V_layer[i] = source_to_target_map[intensity]

# reshape and convert V_layer to 'uint8'
result_V_layer = np.array(np.reshape(flat_V_layer, source_V_layer.shape), dtype='uint8')

# build layers of result image
result_image = np.zeros(source_image.shape)
result_image[:, :, 0] = source_hsv_image[:, :, 0]
result_image[:, :, 1] = source_hsv_image[:, :, 1]
result_image[:, :, 2] = result_image

# convert result to 'uint8'
result_image = np.asarray(result_image, dtype='uint8')

# convert result to RGB
result_image = cv2.cvtColor(result_image, cv2.COLOR_HSV2BGR)

# # to show result image
# show_image(image=result_image, resize_ratio=0.3)

# to export result image
cv2.imwrite(filename="im02.jpg", img=result_image)

# show histogram of result image
show_histogram(result_image)
