import numpy as np
import cv2


def get_channels(image):
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]


def get_flow_image(flow):
    height, width = flow.shape[0:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((height, width, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def apply_flow(img, flow):
    height, width = flow.shape[0:2]
    flow = -flow
    flow[:, :, 0] += np.arange(width)
    flow[:, :, 1] += np.arange(height)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res
