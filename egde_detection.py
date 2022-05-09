import cv2
import numpy as np

img = cv2.imread('0_pred.png')
img2 = cv2.imread('0_image.png')
img2 = img2.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 30, 255, 0)
contour, h = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contour)):
    if cv2.contourArea(contour[i]) > 2000:
        approx = cv2.approxPolyDP(contour[i], 0.005, True)
        img2 = cv2.drawContours(img2, [approx], 0, (255, 255, 255), 5)
cv2.imwrite('cnt.png', img2)