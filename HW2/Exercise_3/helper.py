import cv2
import numpy as np
import math


class Book:
    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4 = x1, y1, x2, y2, x3, y3, x4, y4
        self.f_x1, self.f_y1 = 0, 0
        self.f_x2, self.f_y2 = self.get_width(), 0
        self.f_x3, self.f_y3 = self.get_width(), self.get_height()
        self.f_x4, self.f_y4 = 0, self.get_height()

    def get_width(self):
        return int(math.sqrt((self.y2 - self.y1) ** 2 + (self.x2 - self.x1) ** 2))

    def get_height(self):
        return int(math.sqrt((self.y3 - self.y2) ** 2 + (self.x3 - self.x2) ** 2))

    def get_transform_matrix(self):
        H = np.array([
            [-self.x1, -self.y1, -1, 0, 0, 0, self.x1 * self.f_x1, self.y1 * self.f_x1, self.f_x1],
            [0, 0, 0, -self.x1, -self.y1, -1, self.x1 * self.f_y1, self.y1 * self.f_y1, self.f_y1],

            [-self.x2, -self.y2, -1, 0, 0, 0, self.x2 * self.f_x2, self.y2 * self.f_x2, self.f_x2],
            [0, 0, 0, -self.x2, -self.y2, -1, self.x2 * self.f_y2, self.y2 * self.f_y2, self.f_y2],

            [-self.x3, -self.y3, -1, 0, 0, 0, self.x3 * self.f_x3, self.y3 * self.f_x3, self.f_x3],
            [0, 0, 0, -self.x3, -self.y3, -1, self.x3 * self.f_y3, self.y3 * self.f_y3, self.f_y3],

            [-self.x4, -self.y4, -1, 0, 0, 0, self.x4 * self.f_x4, self.y4 * self.f_x4, self.f_x4],
            [0, 0, 0, -self.x4, -self.y4, -1, self.x4 * self.f_y4, self.y4 * self.f_y4, self.f_y4],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        result_matrix = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        transform_matrix = np.linalg.solve(H, result_matrix)
        return transform_matrix.reshape((3, 3))

    def get_book_image(self, image):
        image_width = image.shape[1]
        image_height = image.shape[0]

        wrapped_image = cv2.warpPerspective(image, self.get_transform_matrix(), (image_height, image_width))
        cropped_image = wrapped_image[:self.get_height(), :self.get_width()]

        return cropped_image
