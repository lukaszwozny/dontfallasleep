import cv2
import numpy as np

class Webcam:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def __init__(self):
        self.winks = 0
        print("init22")

    def testFunc(self):
        print('testf22unc {0}'.format(self.winks))

    def start_detection(self, img, gray_img):
        img_copy = img.copy()
        face = self.detect_face(gray_img=gray_img)

        return img

    @staticmethod
    def detect_face(gray_img):
        return Webcam.face_cascade.detectMultiScale(gray_img, 1.3, 5)

    @staticmethod
    def detect_eyes(gray_roi_img):
        return Webcam.eye_cascade.detectMultiScale(gray_roi_img)

    @staticmethod
    def hconcat(img1, img2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

        vis[:h1, :w1, :3] = img1
        vis[:h2, w1:w1 + w2, :3] = img2

        return vis

    @staticmethod
    def vconcat(img1, img2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)

        vis[:h1, :w1, :3] = img1
        vis[h1:h1 + h2, :w2, :3] = img2

        return vis
