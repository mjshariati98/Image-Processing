import cv2
import numpy as np
import imageio
from scipy.spatial import Delaunay

from HW5.Exercise_2.helper import show_triangles, get_middle_morphing_image

# read images
first_image = cv2.cvtColor(cv2.imread("../resources/q2-first.jpg"), cv2.COLOR_BGR2RGB)
last_image = cv2.cvtColor(cv2.imread("../resources/q2-last.jpg"), cv2.COLOR_BGR2RGB)

first_image_points = np.array([
    # main pic points
    [0, 0], [300, 0], [600, 0], [0, 400], [0, 800], [600, 400], [600, 800], [300, 800],
    # dress
    [0, 770], [600, 705], [264, 740],
    # left eyebrow
    [159, 327], [181, 312], [205, 307], [234, 309], [261, 322],
    # right eyebrow
    [318, 322], [348, 304], [376, 299], [408, 304], [429, 322],
    # right of face
    [471, 341], [474, 393], [495, 404], [469, 443], [464, 487], [456, 518], [432, 564], [384, 586], [349, 609],
    # chin
    [302, 621],
    # left of face
    [249, 612], [204, 592], [182, 570], [161, 543], [142, 514], [130, 443], [128, 405], [128, 356],
    # left eye
    [184, 361], [205, 347], [228, 347], [253, 362], [226, 364], [208, 364],
    # right eye
    [338, 357], [360, 338], [383, 338], [407, 352], [381, 359], [361, 359],
    # nose
    [289, 344], [290, 381], [294, 415], [293, 442], [261, 459], [276, 461], [297, 467], [315, 457], [330, 453],
    # lip
    [218, 492], [230, 497], [247, 492], [268, 490], [268, 500], [292, 489], [295, 500], [326, 485], [328, 496],
    [352, 485], [371, 488], [378, 486], [356, 520], [327, 540], [325, 520], [299, 544], [301, 526], [267, 541],
    [268, 524], [244, 525]
])

last_image_points = np.array([
    # main pic points
    [0, 0], [300, 0], [600, 0], [0, 400], [0, 800], [600, 400], [600, 800], [300, 800],
    # dress
    [0, 690], [600, 590], [290, 680],
    # left eyebrow
    [160, 301], [170, 282], [193, 271], [218, 272], [242, 287],
    # right eyebrow
    [304, 280], [333, 272], [363, 271], [389, 276], [411, 298],
    # right of face
    [462, 329], [465, 370], [514, 364], [464, 415], [458, 456], [431, 492], [398, 514], [357, 546], [317, 571],
    # chin
    [278, 575],
    # left of face
    [231, 561], [201, 534], [177, 509], [159, 473], [152, 442], [147, 410], [146, 377], [153, 338],
    # left eye
    [182, 323], [204, 313], [227, 313], [242, 322], [224, 325], [206, 324],
    # right eye
    [319, 323], [345, 313], [370, 315], [379, 321], [367, 325], [346, 325],
    # nose
    [274, 312], [272, 339], [273, 372], [272, 410], [241, 416], [254, 426], [273, 432], [294, 421], [313, 414],
    # lip
    [210, 460], [222, 459], [231, 456], [257, 456], [258, 465], [274, 455], [275, 464], [301, 452], [300, 461],
    [322, 455], [348, 457], [355, 459], [322, 481], [301, 487], [300, 473], [281, 488], [280, 475], [260, 487],
    [259, 472], [236, 474]
])

# Delaunay algorithm for getting triangles on first image
triangles_delaunay = Delaunay(first_image_points)

show_triangles(first_image, first_image_points, triangles_delaunay)
show_triangles(last_image, last_image_points, triangles_delaunay)

# get triangles array:
# triangles is a list of tuples
# (a, b, c) -> represent a triangle that first vertex is a-th point, second is b-th point and third is c-th point.
triangles = triangles_delaunay.simplices

MIDDLE_IMAGES_COUNT = 20
middle_images = []

for step in range(MIDDLE_IMAGES_COUNT):
    k = step / MIDDLE_IMAGES_COUNT
    middle_image = get_middle_morphing_image(k, first_image, first_image_points,
                                             last_image, last_image_points, triangles)

    # cv2.imwrite("out/pics/" + str(step) + ".jpg", cv2.cvtColor(np.uint8(middle_image), cv2.COLOR_RGB2BGR))
    middle_images.append(middle_image)

# Build GIF image
imageio.mimsave('out/image_morphing.gif', middle_images)
