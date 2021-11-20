import numpy as np
import cv2
from HW1.Exercise_1.helper import gamma_function, logarithm_function, show_image

# read source image
source_image = cv2.imread(filename='../resources/im030.jpg')

# apply gamma function on source image
result_image = np.array(gamma_function(source_image, alpha=0.5), dtype='uint8')

# apply logarithm function on source image
# result_image = np.array(logarithm_function(source_image, alpha=0.6), dtype='uint8')

# # To show result image
# show_image(image=result_image, resize_ratio=0.3)

# To export result image
cv2.imwrite(filename="im01.jpg", img=result_image)
