import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect_face(gray_img):
    return face_cascade.detectMultiScale(gray_img, 1.3, 5)


def detect_eyes(gray_roi_img):
    return eye_cascade.detectMultiScale(gray_roi_img)


def get_detected_img(img, gray_img):
    img_copy = img.copy()
    faces = detect_face(gray_img=gray_img)

    all_eyes_with_height = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_color = img_copy[y:y + h, x:x + w]

        eyes = detect_eyes(gray_roi_img=roi_gray)
        i = 0
        print('face_h: %d', h)
        for (ex, ey, ew, eh) in eyes:
            ab_ey = y + ey
            ab_ex = x + ex
            eye_img = img[ab_ey:ab_ey + eh, ab_ex:ab_ex + ew]
            # show_all_detected_eyes(eye_img, i, img_all_eyes)
            if ey - eh/2 < h / 2:
                all_eyes_with_height.append((eye_img, ab_ey))
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            i += 1

    # if img_all_eyes is not None:
    #     cv2.imshow('eyes', img_all_eyes)
    # print(len(all_eyes))
    show_all_detected_eyes(all_eyes_with_height)
    return img_copy


def is_open(param):
    pass


def show_all_detected_eyes(eyes):
    eyes_all_img = None
    if len(eyes) == 0:
        return

    i = 0
    for eye in eyes:
        if i == 0:
            eyes_all_img = eye[0]
        else:
            eyes_all_img = vconcat(eye[0], eyes_all_img)
        i += 1
    cv2.imshow('eyes', eyes_all_img)


def is_eye_open(eye_img):
    width, height = eye_img.shape[:2]
    blank_image = eye_img.copy()
    # cv2.normalize(eye_img, eye_img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    for y in range(height):
        for x in range(width):
            # Get RGB
            bgr = eye_img[x][y]
            r = int(bgr[2])
            g = int(bgr[1])
            b = int(bgr[0])

            # Check if white
            min_value = 100
            min_value_r = 80
            max_dif_rg = 15
            max_dif_b = 40
            if r > min_value_r and g > min_value and b > min_value:
                if abs(r - g) < max_dif_rg and b - r < max_dif_b and b - g < max_dif_b:
                    blank_image[x][y] = (0, 0, 255)

            # Check if black
            max_value_rg = 50
            max_value_b = 60
            max_dif_rg = 15
            max_dif_b = 40
            if r < max_value_rg and g < max_value_rg and b < max_value_b:
                if abs(r - g) < max_dif_rg and b - r < max_dif_b and b - g < max_dif_b:
                    blank_image[x][y] = (0, 255, 0)

    return blank_image


def vconcat(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    vis[:h1, :w1, :3] = img1
    vis[:h2, w1:w1 + w2, :3] = img2

    return vis


def hconcat(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)

    vis[:h1, :w1, :3] = img1
    vis[h1:h1 + h2, :w2, :3] = img2

    return vis
