import cv2
import numpy as np
import webcam

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img, 1)  # flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get all images
    img_detected = webcam.get_detected_img(img=img, gray_img=gray)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    # cv2.imshow('lap', sobelx)

    # connect all images
    org_det_img = cv2.hconcat((img, img_detected))
    org_det_img2 = cv2.hconcat((sobelx, sobely))
    # all_img = cv2.vconcat((org_det_img, org_det_img2))

    # Show all images
    cv2.imshow('img', org_det_img)
    # cv2.imshow('img2', org_det_img2)

    # Close when ESC pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# # Load an color image in grayscale
# open_img = cv2.imread('open.jpg')
# open_img_n = open_img.copy()
# open_img2 = cv2.imread('open_2.jpg')
# open_2_img_n = open_img2.copy()
# cv2.normalize(open_img2, open_2_img_n, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
# cv2.normalize(open_img, open_img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
# close_img = cv2.imread('close.jpg')
#
# open_check = webcam.is_eye_open(open_img)
# open2_check = webcam.is_eye_open(open_img2)
# close_check = webcam.is_eye_open(close_img)
#
# # Concat
# opens = webcam.vconcat(open_img, open_check)
# closed = webcam.vconcat(close_img, close_check)
# set1 = webcam.hconcat(opens, closed)
#
# opens2 = webcam.vconcat(open_img2, open2_check)
#
# cv2.imshow('set1', set1)
# cv2.imshow('opens2', opens2)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
