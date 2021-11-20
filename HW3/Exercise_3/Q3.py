# the idea of this approach:
# https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import skimage.color as color

from HW3.Exercise_3.helper import grab_cut_with_rectangle, create_special_mask_for_image_1, grab_cut_with_mask, \
    load_images, create_special_mask_for_image_2

images = load_images()
# i=0 --> first image (im053)
# i=1 --> second image (im054)
for i in range(0, 2):
    image = images[i]
    height, width = image.shape[0:2]

    # use slic for over-segmentation
    segments = slic(image, n_segments=500, convert2lab=True, sigma=5)

    cluster_image = color.label2rgb(segments, image, kind='avg')

    cluster_image = np.array((cluster_image / cluster_image.max()) * 255, dtype='uint8')
    over_seg = mark_boundaries(cluster_image, segments)
    plt.imsave("out/over-segmentation" + str(i + 1) + ".jpg", over_seg)

    if i == 0:
        rect = (245, 70, 400, 500)
    else:
        rect = (350, 280, 532, 240)

    mask, mask2, background_model, foreground_model = grab_cut_with_rectangle(cluster_image, rect, width, height)

    first_grab_cut_img = image * mask2[:, :, np.newaxis]
    plt.imsave("out/first_grab_cut_image" + str(i + 1) + ".jpg", first_grab_cut_img)

    if i == 0:
        mask = create_special_mask_for_image_1(mask, width, height)
    else:
        mask = create_special_mask_for_image_2(mask, width, height)

    mask = grab_cut_with_mask(cluster_image, mask, background_model, foreground_model)

    result = image * mask[:, :, np.newaxis]
    if i == 0:
        plt.imsave("out/im06.jpg", result)
    else:
        plt.imsave("out/im07.jpg", result)
