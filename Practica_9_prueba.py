import cv2
import numpy as np

image = cv2.imread("imagen.jpg")
img = cv2.imread("imagen.jpg")
template1 = cv2.imread("template1.jpg")
template2 = cv2.imread("template2.jpg")
template3 = cv2.imread("template3.jpg")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template1_gray = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template2_gray = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
template3_gray = cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)

res1 = cv2.matchTemplate(image_gray, template1_gray, cv2.TM_SQDIFF)
res2 = cv2.matchTemplate(image_gray, template2_gray, cv2.TM_SQDIFF)
res3 = cv2.matchTemplate(image_gray, template3_gray, cv2.TM_SQDIFF)

min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)

print(min_val1, max_val1, min_loc1, max_loc1)
print(min_val2, max_val2, min_loc2, max_loc2)
print(min_val3, max_val3, min_loc3, max_loc3)

x1, y1 = min_loc1
x2, y2 = min_loc1[0] + template1.shape[1], min_loc1[1] + template1.shape[0]
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

x3, y3 = min_loc2
x4, y4 = min_loc2[0] + template2.shape[1], min_loc2[1] + template2.shape[0]
cv2.rectangle(image, (x3, y3), (x4, y4), (0, 255, 0), 3)

x5, y5 = min_loc3
x6, y6 = min_loc3[0] + template3.shape[1], min_loc3[1] + template3.shape[0]
cv2.rectangle(image, (x5, y5), (x6, y6), (0, 255, 0), 3)

cv2.imshow('Original',img)
cv2.imshow("Imagen", image)
cv2.imshow("Template1", template1)
cv2.imshow("Template2", template2)
cv2.imshow("Template3", template3)
cv2.waitKey(0)
cv2.destroyAllWindows()
