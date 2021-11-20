import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image(image, resize_ratio):
    resized_image = cv2.resize(src=image, dsize=None, fx=resize_ratio, fy=resize_ratio)
    cv2.imshow(winname="image", mat=resized_image)
    cv2.waitKey(0)


def show_triangles(image, points, triangles):
    plt.imshow(image)
    plt.triplot(points[:, 0], points[:, 1], triangles.simplices.copy())
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()


def get_middle_morphing_image(k, first_image, first_image_points, last_image, last_image_points, triangles):
    middle_image = np.zeros(first_image.shape)
    for triangle in triangles:
        first_image_triangle = [first_image_points[triangle[0]], first_image_points[triangle[1]],
                                first_image_points[triangle[2]]]
        # for a triangle, first_image_triangle is something like: [array([300,   0]), array([181, 312]), array([0, 0])]
        last_image_triangle = [last_image_points[triangle[0]], last_image_points[triangle[1]],
                               last_image_points[triangle[2]]]

        # get middle image correspond triangle
        middle_image_triangle = get_middle_triangle(k, first_image_triangle, last_image_triangle)

        # get min boundary rectangle for first, last and middle triangles
        first_image_rectangle = cv2.boundingRect(np.float32([first_image_triangle]))
        last_image_rectangle = cv2.boundingRect(np.float32([last_image_triangle]))
        middle_image_rectangle = cv2.boundingRect(np.float32([middle_image_triangle]))

        # change dimension of rectangles: change top-left corner coordinate of rectangle to (0,0)
        first_image_rectangle_new_dimension = change_rectangle_dimension(first_image_triangle, first_image_rectangle)
        last_image_rectangle_new_dimension = change_rectangle_dimension(last_image_triangle, last_image_rectangle)
        middle_image_rectangle_new_dimension = change_rectangle_dimension(middle_image_triangle, middle_image_rectangle)

        # get rectangle part of images
        x_first_image_rectangle = first_image_rectangle[0]
        y_first_image_rectangle = first_image_rectangle[1]
        width_first_image_rectangle = first_image_rectangle[2]
        height_first_image_rectangle = first_image_rectangle[3]
        first_image_rectangle_part = get_rectangle_part_of_image(first_image,
                                                                 x_first_image_rectangle,
                                                                 y_first_image_rectangle,
                                                                 width_first_image_rectangle,
                                                                 height_first_image_rectangle)
        x_last_image_rectangle = last_image_rectangle[0]
        y_last_image_rectangle = last_image_rectangle[1]
        width_last_image_rectangle = last_image_rectangle[2]
        height_last_image_rectangle = last_image_rectangle[3]
        last_image_rectangle_part = get_rectangle_part_of_image(last_image,
                                                                x_last_image_rectangle,
                                                                y_last_image_rectangle,
                                                                width_last_image_rectangle,
                                                                height_last_image_rectangle)

        # get output triangle width and height
        main_images_width = first_image.shape[1]
        main_images_height = first_image.shape[0]
        output_width, output_height = get_output_rectangle_size(main_images_width,
                                                                main_images_height,
                                                                middle_image_rectangle)

        # warp rectangle part of first and last images
        warped_rectangle_part_of_first_image = find_and_warp_affine_transform(first_image_rectangle_part,
                                                                              first_image_rectangle_new_dimension,
                                                                              middle_image_rectangle_new_dimension,
                                                                              output_width, output_height)
        warped_rectangle_part_of_last_image = find_and_warp_affine_transform(last_image_rectangle_part,
                                                                             last_image_rectangle_new_dimension,
                                                                             middle_image_rectangle_new_dimension,
                                                                             output_width, output_height)

        # get middle image rectangle part using warped first and last images and k ratio
        middle_rectangle_part = (1.0 - k) * warped_rectangle_part_of_first_image + \
                                k * warped_rectangle_part_of_last_image

        # build a mask with output size: inside of triangle = 1, outside of triangle = 0
        mask = np.zeros((output_height, output_width, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(middle_image_rectangle_new_dimension), (255, 255, 255), 16, 0)
        mask = np.vectorize(lambda x: 1 if x == 255 else 0)(mask)

        # finally, build middle image!
        middle_image = fill_rectangle_in_middle_image(middle_image, middle_image_rectangle, middle_rectangle_part, mask)

    return np.uint8(middle_image)


def get_middle_triangle(k, first_image_triangle, last_image_triangle):
    """
    According to first and last triangles, calculates the coordinates of triangle in middle image

    :param k: ratio step / number of middle images
    :param first_image_triangle: triangle in first image
    :param last_image_triangle:  correspond triangle in last image
    :return: triangle in middle image
    """
    middle_triangle = []
    for vertex in range(3):
        middle_vertex = [k * last_image_triangle[vertex][0] + (1 - k) * first_image_triangle[vertex][0],
                         k * last_image_triangle[vertex][1] + (1 - k) * first_image_triangle[vertex][1]]
        middle_triangle.append(middle_vertex)

    return middle_triangle


def change_rectangle_dimension(triangle, rectangle):
    """
    Change dimension of rectangles to top-left corner coordinate of rectangle be (0,0)

    :param triangle:
    :param rectangle:
    :return:
    """
    rectangle_new_dimension = []
    for i in range(0, 3):
        rectangle_new_dimension.append(((triangle[i][0] - rectangle[0]), (triangle[i][1] - rectangle[1])))

    return rectangle_new_dimension


def get_rectangle_part_of_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]


def get_output_rectangle_size(main_images_width, main_images_height, middle_image_rectangle):
    width = middle_image_rectangle[2]
    if middle_image_rectangle[0] + middle_image_rectangle[2] > main_images_width:
        width = main_images_width - middle_image_rectangle[0]

    height = middle_image_rectangle[3]
    if middle_image_rectangle[1] + middle_image_rectangle[3] > main_images_height:
        height = main_images_height - middle_image_rectangle[1]

    return width, height


def find_and_warp_affine_transform(initial_pic, initial_points, final_points, width, height):
    """
    According to points of rectangle in first_image and points of rectangle in middle_image, find the Affine transform.
    Then, transform first_image with that func to get middle image.
    :param initial_pic: source pic
    :param initial_points: source points
    :param final_points:  destination points
    :param width: output width
    :param height: output height
    :return: pic transformed
    """
    affine_transform = cv2.getAffineTransform(np.float32(initial_points), np.float32(final_points))
    final_pic = cv2.warpAffine(initial_pic, affine_transform, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    return final_pic


def fill_rectangle_in_middle_image(middle_image, middle_image_rectangle, middle_rectangle_part, mask):
    """
    Filling triangle in middle image
    Have a rectangle and a triangle in that rectangle!
    Outside of triangle must don't change, so we multiple middle_image to (1-mask)

    :param middle_image: the result pic
    :param middle_image_rectangle: the rectangle correspond to this middle image
    :param middle_rectangle_part: part of middle image that rectangle is there
    :param mask: a mask that inside of triangle is 1 and outside of it is 0
    :return: the middle image (result pic)
    """
    x = middle_image_rectangle[0]  # x_middle_rect
    y = middle_image_rectangle[1]  # y_middle_rect
    width = middle_image_rectangle[2]  # width_middle_rect
    height = middle_image_rectangle[3]  # height_middle_rect

    middle_image[y:y + height, x:x + width] = \
        middle_rectangle_part * mask + middle_image[y:y + height, x:x + width] * (1 - mask)

    return middle_image
