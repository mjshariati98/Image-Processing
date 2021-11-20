import cv2
import numpy as np
import imageio
from HW4.Exercise_5.helper import get_flow_image, apply_flow, get_channels

# Load images
frame_1 = cv2.imread("../resources/01.png")
frame_2 = cv2.imread("../resources/02.png")

# Separate channels
B_1, G_1, R_1 = get_channels(frame_1)
B_2, G_2, R_2 = get_channels(frame_2)

# Optical Flow on each channel
flow_on_B_channel = cv2.calcOpticalFlowFarneback(prev=B_1, next=B_2, flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
flow_on_G_channel = cv2.calcOpticalFlowFarneback(prev=G_1, next=G_2, flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
flow_on_R_channel = cv2.calcOpticalFlowFarneback(prev=R_1, next=R_2, flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# save flow images
flow_B = get_flow_image(flow_on_B_channel)
cv2.imwrite("out/flow_B.jpg", flow_B)
flow_G = get_flow_image(flow_on_G_channel)
cv2.imwrite("out/flow_G.jpg", flow_G)
flow_R = get_flow_image(flow_on_R_channel)
cv2.imwrite("out/flow_R.jpg", flow_R)

# apply flow on channels and save result
result_image = np.zeros(frame_1.shape)
result_image[:, :, 0] = apply_flow(B_1, flow_on_B_channel)
result_image[:, :, 1] = apply_flow(G_1, flow_on_G_channel)
result_image[:, :, 2] = apply_flow(R_1, flow_on_R_channel)
cv2.imwrite("out/im5.jpg", result_image)

# Build GIF image
images = [result_image, frame_2]
imageio.mimsave('out/flow.gif', images)
