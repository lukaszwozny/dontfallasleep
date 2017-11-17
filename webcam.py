import cv2
import numpy as np


class Webcam:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def __init__(self):
        print("init")

    def testFunc(self):
        print('testfunc')

    wings = 0
    is_open_flag = True

    x_diff_max = 0
    y_diff_max = 0

    def detect_face(gray_img):
        return Webcam.face_cascade.detectMultiScale(gray_img, 1.3, 5)

    def detect_eyes(gray_roi_img):
        return Webcam.eye_cascade.detectMultiScale(gray_roi_img)

    def get_detected_img(img, gray_img):
        img_copy = img.copy()
        faces = Webcam.detect_face(gray_img=gray_img)

        all_eyes = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_color = img_copy[y:y + h, x:x + w]

            eyes = Webcam.detect_eyes(gray_roi_img=roi_gray)
            for (ex, ey, ew, eh) in eyes:
                ab_ey = y + ey
                ab_ex = x + ex
                eye_img = img[ab_ey:ab_ey + eh, ab_ex:ab_ex + ew]
                # show_all_detected_eyes(eye_img, i, img_all_eyes)
                if ey < h / 2:
                    all_eyes.append(eye_img)
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # if img_all_eyes is not None:
        #     cv2.imshow('eyes', img_all_eyes)
        # print(len(all_eyes))
        Webcam.show_all_detected_eyes(all_eyes)
        return img_copy

    def show_all_detected_eyes(eyes):
        eyes_all_img = None
        if len(eyes) == 0:
            return

        i = 0
        opens = 0
        for eye in eyes:
            is_open, eye = Webcam.is_eye_open(eye, (0, 0, 0))
            if is_open:
                opens += 1
            if i == 0:
                eyes_all_img = eye
            else:
                eyes_all_img = Webcam.vconcat(eye, eyes_all_img)
            i += 1
        cv2.imshow('eyes', eyes_all_img)

        open_green = np.zeros((40, 40, 3), np.uint8)
        for y in range(40):
            for x in range(40):
                open_green[x][y] = (255, 255, 255)
        close_black = np.zeros((40, 40, 3), np.uint8)

        if opens == 2:
            cv2.imshow('is_open', open_green)
            if not Webcam.is_open_flag:
                Webcam.is_open_flag = True
                Webcam.wings += 1
                print('wings: {0}'.format(Webcam.wings))
        elif opens == 0:
            cv2.imshow('is_open', close_black)
            if Webcam.is_open_flag:
                Webcam.is_open_flag = False

    def is_eye_open(eye_img, min_color):
        is_open = False

        black_pixels = 0
        black_pixels_x = 0
        black_pixels_y = 0
        black_middle_x = 0
        black_middle_y = 0

        white_pixels = 0
        white_pixels_x = 0
        white_pixels_y = 0
        white_middle_x = 0
        white_middle_y = 0

        height, width = eye_img.shape[:2]
        blank_image = eye_img.copy()

        for y in range(height):
            for x in range(width):
                # Get RGB
                bgr = eye_img[y][x]
                r = int(bgr[2])
                g = int(bgr[1])
                b = int(bgr[0])

                # Check if white
                min_white = (177, 142, 152)
                if r > min_white[0] and g > min_white[1] and b > min_white[2]:
                    if abs(height / 2 - y) < height / 2 * 0.5 \
                            and abs(width / 2 - x) < width / 2 * 0.75:
                        blank_image[y][x] = (0, 0, 255)
                        white_pixels += 1
                        white_pixels_x += x
                        white_pixels_y += y

                # Check if black
                max_black = (122, 82, 92)
                if r < max_black[0] and g < max_black[1] and b < max_black[2]:
                    if abs(height / 2 - y) < height / 2 * 0.3 \
                            and abs(width / 2 - x) < width / 2 * 0.30:
                        blank_image[x][y] = (0, 255, 0)
                        black_pixels += 1
                        black_pixels_x += x
                        black_pixels_y += y

        middle_x = width / 2
        middle_y = height / 2
        n_pixels = width * height
        if white_pixels > 0:
            # check is white in the middle
            white_pixels_x = white_pixels_x / white_pixels
            white_pixels_y = white_pixels_y / white_pixels
            white_x_diff = abs(white_pixels_x - middle_x) / middle_x * 100
            white_y_diff = abs(white_pixels_y - middle_y) / middle_y * 100
            white_fill = white_pixels / n_pixels * 100
            if white_x_diff > Webcam.x_diff_max:
                Webcam.x_diff_max = white_x_diff
            if white_y_diff > Webcam.y_diff_max:
                Webcam.y_diff_max = white_y_diff

        if black_pixels > 0:
            # check is black in the middle
            black_pixels_x = black_pixels_x / black_pixels
            black_pixels_y = black_pixels_y / black_pixels
            black_x_diff = abs(black_pixels_x - middle_x) / middle_x * 100
            black_y_diff = abs(black_pixels_y - middle_y) / middle_y * 100
            black_fill = black_pixels / n_pixels * 100
            if black_fill > 2:
                is_open = True

        return is_open, blank_image

    def hconcat(img1, img2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

        vis[:h1, :w1, :3] = img1
        vis[:h2, w1:w1 + w2, :3] = img2

        return vis

    def vconcat(img1, img2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)

        vis[:h1, :w1, :3] = img1
        vis[h1:h1 + h2, :w2, :3] = img2

        return vis
