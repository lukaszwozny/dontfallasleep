import cv2
import numpy as np
from webcam import Webcam
from utils import Webcam as WW

web = WW()


def from_webcam():
    w = Webcam()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.flip(img, 1)  # flip vertically
        img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # get all images
        img_detected = w.get_detected_img(img=img, gray_img=gray)
        # cv2.imshow('lap', sobelx)

        # connect all images
        org_det_img = w.hconcat(img, img_detected)
        # all_img = cv2.vconcat((org_det_img, org_det_img2))

        # Show all images
        cv2.imshow('img', org_det_img)

        # Close when ESC pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == 32:
            cv2.imwrite('face.jpg', org_det_img)

    cap.release()
    cv2.destroyAllWindows()


def from_video_file(filename):
    webcam = WW()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        # img = cv2.flip(img, 1)  # flip vertically
        img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # get all images
        img_detected = webcam.start_detection(img=img, gray_img=gray)
        # cv2.imshow('lap', sobelx)

        # connect all images
        org_det_img = webcam.hconcat(img, img_detected)
        # all_img = cv2.vconcat((org_det_img, org_det_img2))

        # Show all images
        cv2.imshow('img', org_det_img)

        # Close when ESC pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            return
        elif k == 32:
            cv2.imwrite('face.jpg', org_det_img)

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


def from_images():
    w = Webcam
    # Load test images
    open_img = cv2.imread('i//open.jpg')
    close_img = cv2.imread('i//close.jpg')
    openL_img = cv2.imread('i//open_L.jpg')
    openR_img = cv2.imread('i//open_R.jpg')

    # resize for better view
    multiplier = 5
    openL_img = cv2.resize(openL_img, (0, 0), fx=multiplier, fy=multiplier)
    openR_img = cv2.resize(openR_img, (0, 0), fx=multiplier, fy=multiplier)

    # openLnorm_img = None
    # cv2.normalize(my_open_img, my_open_img, 0, 255, cv2.NORM_MINMAX)
    open_img_n = open_img.copy()
    cv2.normalize(open_img, open_img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    cv2.normalize(openL_img, openL_img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    cv2.normalize(openR_img, openR_img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)

    minR = 127
    minG = 127
    minB = 127
    while 1:
        min_color = (minR, minG, minB)
        open_check = w.is_eye_open(open_img)
        close_check = w.is_eye_open(close_img)
        # openL_check = webcam.is_eye_open_adj(openL_img, min_color)
        is_openR_check, openR_check = webcam.is_eye_open(openR_img, min_color)

        # Concat 1
        opens = w.hconcat(open_img, open_check)
        closed = w.hconcat(close_img, close_check)
        set1 = w.vconcat(opens, closed)

        # Concat 2
        # opensL = webcam.hconcat(openL_img, openL_check)
        opensR = w.hconcat(openR_img, openR_check)
        # set2 = webcam.vconcat(opensL, opensR)

        cv2.imshow('set1', set1)
        cv2.imshow('set2', opensR)

        # Close when ESC pressed
        value = 5
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            return
        if k == ord('1'):
            minR -= value
            if minR < 0:  # R--
                minR = 0
            print((minR, minG, minB))
        if k == ord('2'):  # G--
            minG -= value
            if minG < 0:
                minG = 0
            print((minR, minG, minB))
        if k == ord('3'):  # B--
            minB -= value
            if minB < 0:
                minB = 0
            print((minR, minG, minB))
        if k == ord('4'):  # R++
            minR += value
            if minR > 255:
                minR = 255
            print((minR, minG, minB))
        if k == ord('5'):  # G++
            minG += value
            if minG > 255:
                minG = 255
            print((minR, minG, minB))
        if k == ord('6'):  # B++
            minB += value
            if minB > 255:
                minB = 255
            print((minR, minG, minB))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    web.testFunc()

    from_video_file('woman.mp4')
    # from_webcam()
    # from_images()
