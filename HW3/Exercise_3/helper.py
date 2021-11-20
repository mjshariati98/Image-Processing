import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage import io


def load_images():
    images = []
    image1 = img_as_float(io.imread("../resources/im053.jpg"))
    images.append(image1)
    image2 = img_as_float(io.imread("../resources/im054.jpg"))
    images.append(image2)

    return images


def grab_cut_with_rectangle(cluster_image, rect, width, height):
    mask = np.zeros((height, width), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(cluster_image, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask, mask2, background_model, foreground_model


def create_special_mask_for_image_1(mask, width, height):
    # remove these from mask
    new_mask = np.zeros((height, width), np.uint8)
    new_mask = cv2.line(new_mask, (620, 285), (620, 200), (255, 255, 255), 40)
    new_mask = cv2.line(new_mask, (500, 489), (500, 315), (255, 255, 255), 20)
    new_mask = cv2.line(new_mask, (290, 150), (290, 240), (255, 255, 255), 20)
    new_mask = cv2.line(new_mask, (560, 100), (560, 200), (255, 255, 255), 40)
    new_mask = cv2.line(new_mask, (600, 370), (550, 290), (255, 255, 255), 50)
    mask[new_mask == 255] = 0

    # add this to mask
    new_mask = np.zeros((height, width), np.uint8)
    new_mask = cv2.line(new_mask, (400, 260), (400, 320), (255, 255, 255), 70)
    mask[new_mask == 255] = 1

    return mask


def create_special_mask_for_image_2(mask, width, height):
    # remove these from mask
    new_mask = np.zeros((height, width), np.uint8)
    new_mask = cv2.line(new_mask, (810, 410), (880, 450), (255, 255, 255), 30)
    new_mask = cv2.line(new_mask, (350, 450), (450, 440), (255, 255, 255), 45)
    new_mask = cv2.line(new_mask, (530, 450), (550, 465), (255, 255, 255), 20)
    new_mask = cv2.line(new_mask, (450, 480), (380, 400), (255, 255, 255), 20)
    new_mask = cv2.line(new_mask, (450, 480), (550, 480), (255, 255, 255), 20)
    mask[new_mask == 255] = 0

    # add this to mask
    new_mask = np.zeros((height, width), np.uint8)
    new_mask = cv2.line(new_mask, (750, 345), (820, 305), (255, 255, 255), 15)
    mask[new_mask == 255] = 1

    return mask


def grab_cut_with_mask(cluster_image, mask, background_model, foreground_model):
    mask, background_model, foreground_model = cv2.grabCut(cluster_image, mask, None, background_model,
                                                           foreground_model, 5, cv2.GC_INIT_WITH_MASK)
    plt.imshow(mask)
    plt.show()

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    return mask
