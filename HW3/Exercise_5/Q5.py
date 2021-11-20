from skimage.segmentation import mark_boundaries
import skimage.color as color
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np

from HW3.Exercise_5.helper import slic_oversegmentation, merge_super_pixels

RADIUS = 35
THRESHOLD = 80

image = img_as_float(io.imread("../resources/im023.jpg"))

# only birds for faster compute
image = image[1400:2600, :, :]

# sharp image
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpen_image = cv2.filter2D(image, -1, kernel)

# SLIC over-segmentation
segments = slic_oversegmentation(sharpen_image)

segmented_image = color.label2rgb(segments, image, kind='avg')
final_over_segmentation = mark_boundaries(segmented_image, segments)
final_over_segmentation = np.array((final_over_segmentation / final_over_segmentation.max()) * 255, 'uint8')
cv2.imwrite("out/over-segmentation.jpg", final_over_segmentation)

# birds points
birds = np.array(
    [[275, 910], [405, 905], [550, 850], [715, 850], [855, 820], [1070, 770], [1270, 755], [1570, 710],
     [1960, 635], [2115, 620], [2255, 585], [2395, 565], [2505, 535], [2670, 520], [3030, 440],
     [3135, 380], [3285, 350], [3445, 335], [3815, 250], [4065, 195], [4320, 150]])

# show birds point
plt.scatter(birds[:, 0], birds[:, 1])
plt.imshow(image)
plt.show()

# merge super pixels
segments = merge_super_pixels(birds, segments, segmented_image, RADIUS, THRESHOLD)

final_over_segmentation = mark_boundaries(image, segments)
final_over_segmentation = np.array((final_over_segmentation / final_over_segmentation.max()) * 255, 'uint8')
cv2.imwrite("out/im09.jpg", final_over_segmentation)

label_image = color.label2rgb(segments, image, kind='avg')
label_image = np.array((label_image / label_image.max()) * 255, dtype='uint8')
cv2.imwrite("out/im09_label_image.jpg", label_image)


# 8.33870792388916  sec for over-segmentation
# 13.709674596786499  min for merge super pixels