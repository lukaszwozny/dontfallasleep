import cv2
import numpy as np
import webcam

# cap = cv2.VideoCapture(0)
#
# while 1:
#     ret, img = cap.read()
#     img = cv2.flip(img, 1)  # flip vertically
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # get all images
#     img_detected = webcam.get_detected_img(img=img, gray_img=gray)
#     sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
#     sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#     # cv2.imshow('lap', sobelx)
#
#     # connect all images
#     org_det_img = cv2.hconcat((img, img_detected))
#     org_det_img2 = cv2.hconcat((sobelx, sobely))
#     # all_img = cv2.vconcat((org_det_img, org_det_img2))
#
#     # Show all images
#     cv2.imshow('img', org_det_img)
#     # cv2.imshow('img2', org_det_img2)
#
#     # Close when ESC pressed
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# Load an color image in grayscale
open_img = cv2.imread('open.jpg')
close_img = cv2.imread('close.jpg')
cv2.imshow('open', open_img)
cv2.imshow('close', close_img)

width, height = open_img.shape[:2]
print(width, height)
print(width, height)
open_check = webcam.is_eye_open(open_img)
close_check = webcam.is_eye_open(close_img)

cv2.imshow('open_check', open_check)
cv2.imwrite('open_check.jpg', open_check)
cv2.imshow('close_check', close_check)
cv2.imwrite('close_check.jpg', close_check)

cv2.waitKey(0)
cv2.destroyAllWindows()
