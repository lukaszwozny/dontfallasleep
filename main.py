import cv2
import numpy as np
import time

from my_dlib import start_dlib
from color_detection import Webcam
from cv_utils import display_text, show_fps, show_times

webcam = Webcam()


def show_blinks(img, winks):
    text = 'Blinks: {0}'.format(winks)
    margin = 5
    position = (-1, -1)
    display_text(img=img, text=text, position=position, margin=margin)


def from_webcam():
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
        img_detected = webcam.start_detection(img=img, gray_img=gray)
        # cv2.imshow('lap', sobelx)

        # connect all images
        org_det_img = Webcam.hconcat(img, img_detected)
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
    # cap = cv2.VideoCapture(0)
    video = cv2.VideoCapture(filename)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(video_fps))

    # Start time
    start_time = time.time()
    x = 0.1  # displays the frame rate every 0.1 second
    counter = 0
    fps = 0

    while video.isOpened():
        ret, img = video.read()
        if not ret:
            break
        # img = cv2.flip(img, 1)  # flip vertically
        img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # get all images
        img_detected = webcam.start_detection(img=img, gray_img=gray)
        # cv2.imshow('lap', sobelx)

        counter += 1
        if (time.time() - start_time) > x:
            fps = counter / (time.time() - start_time)
            counter = 0
            start_time = time.time()
        show_fps(img, fps)
        show_blinks(img, webcam.blinks)

        # show_times(img, webcam.open_frames, webcam.close_frames, video_fps)
        show_times(img, webcam.open_frames / video_fps, webcam.get_average_open_time(video_fps))

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
            k = 0
            while k != 32:
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    return
        elif k == ord('s'):
            cv2.imwrite('face.jpg', org_det_img)

    cv2.waitKey(0)
    video.release()
    cv2.destroyAllWindows()


def from_images():
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
        is_open_open, open_check = webcam.is_eye_open(open_img)
        is_open_close, close_check = webcam.is_eye_open(close_img)
        # is_openL_open, openL_check = webcam.is_eye_open_adj(openL_img, min_color)
        is_openR_open, openR_check = webcam.is_eye_open(openR_img)

        # Concat 1
        opens = Webcam.hconcat(open_img, open_check)
        closed = Webcam.hconcat(close_img, close_check)
        set1 = Webcam.vconcat(opens, closed)

        # Concat 2
        # opensL = Webcam.hconcat(openL_img, openL_check)
        opensR = Webcam.hconcat(openR_img, openR_check)
        # set2 = Webcam.vconcat(opensL, opensR)

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
    from_video_file('vids/woman.mp4')
    # from_video_file('vids/woman_2.mp4')
    # start_dlib()
    # start_dlib(filename='vids/woman.mp4')
    # start_dlib(filename='vids/woman_2.mp4')
    # from_webcam()
    # from_images()
