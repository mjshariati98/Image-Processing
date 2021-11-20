import cv2
from HW2.Exercise_3.helper import Book

main_image = cv2.imread(filename="../resources/Books.jpg")

# book-1
x1, y1, x2, y2, x3, y3, x4, y4 = 666, 214, 602, 398, 318, 292, 382, 110
book1 = Book(x1, y1, x2, y2, x3, y3, x4, y4)

print("Transform matrix for book1:")
print(book1.get_transform_matrix())
print("===============================")
result_book1 = book1.get_book_image(main_image)
cv2.imwrite(filename="out/Q3_book1.jpg", img=result_book1)

# book-2
x1, y1, x2, y2, x3, y3, x4, y4 = 364, 742, 154, 710, 206, 430, 410, 468
book2 = Book(x1, y1, x2, y2, x3, y3, x4, y4)

print("Transform matrix for book2:")
print(book2.get_transform_matrix())
print("===============================")
result_book2 = book2.get_book_image(main_image)
cv2.imwrite(filename="out/Q3_book2.jpg", img=result_book2)

# book-3
x1, y1, x2, y2, x3, y3, x4, y4 = 814, 970, 610, 1104, 424, 800, 622, 668
book3 = Book(x1, y1, x2, y2, x3, y3, x4, y4)

print("Transform matrix for book3:")
print(book3.get_transform_matrix())
print("===============================")
result_book3 = book3.get_book_image(main_image)
cv2.imwrite(filename="out/Q3_book3.jpg", img=result_book3)
