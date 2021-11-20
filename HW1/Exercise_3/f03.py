import numpy as np
import cv2
from HW1.Exercise_3.helper import show_image, get_height_and_width, get_BGR_channels, find_match_B_and_G_channel, \
    fix_G_channel, find_match_B_and_R_channel, fix_R_channel

# read source image
source_image = cv2.imread(filename='../resources/BW.tif')

# get height and width of source image
source_height, source_width = get_height_and_width(image=source_image)

# split source image to its channels
B_channel, G_channel, R_channel = get_BGR_channels(image=source_image)

# build result image az zero matrix
result_width = source_width
result_height = source_height // 3
result_image = np.zeros((source_height // 3, source_width, 3))

# fix one channel
result_image[:, :, 0] = B_channel

# find best match of B and G channel and fix it on result image
best_i, best_j = find_match_B_and_G_channel(source_image=source_image, result_image=result_image, L=2)
result_image = fix_G_channel(result_image=result_image, G_channel=G_channel, best_i=best_i, best_j=best_j)

# find best match of B and R channel and fix it on result image
best_i, best_j = find_match_B_and_R_channel(source_image=source_image, result_image=result_image, L=2)
result_image = fix_R_channel(result_image=result_image, R_channel=R_channel, best_i=best_i, best_j=best_j)

# convert element types of result image to 'uint8'
result_image = np.array(result_image, dtype='uint8')

# # To show result image
# show_image(image=result_image, resize_ratio=0.2)

# To export result image
cv2.imwrite(filename="im03.jpg", img=result_image)
