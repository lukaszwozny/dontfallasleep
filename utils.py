import cv2
import numpy as np

class Webcam:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def __init__(self):
        self.winks = 0
        self.is_open = True

    def start_detection(self, img, gray_img):
        img_copy = img.copy()
        faces = self.detect_face(gray_img=gray_img)

        all_eyes = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_color = img_copy[y:y + h, x:x + w]

            eyes = self.detect_eyes(gray_roi_img=roi_gray)
            for (ex, ey, ew, eh) in eyes:
                ab_ey = y + ey
                ab_ex = x + ex
                eye_img = img[ab_ey:ab_ey + eh, ab_ex:ab_ex + ew]
                if ey < h / 2:
                    all_eyes.append(eye_img)
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        self.show_all_detected_eyes(all_eyes)
        return img_copy


    def show_all_detected_eyes(self, eyes):
        eyes_all_img = None
        if len(eyes) == 0:
            return

        i = 0
        opens = 0
        for eye in eyes:
            is_open, eye = self.is_eye_open(eye)
            if is_open:
                opens += 1
            if i == 0:
                eyes_all_img = eye
            else:
                eyes_all_img = self.vconcat(eye, eyes_all_img)
            i += 1
        cv2.imshow('eyes', eyes_all_img)

        if opens == 2:
            if not self.is_open:
                self.is_open = True
                self.winks += 1
        elif opens == 0:
            if self.is_open:
                self.is_open = False

    def is_eye_open(self, eye_img):
        is_open = False

        black_pixels = 0
        black_pixels_x = 0
        black_pixels_y = 0

        white_pixels = 0
        white_pixels_x = 0
        white_pixels_y = 0

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


    def is_eye_open2(self, eye_img):
        is_open = False
        hsv_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2HSV)

        black_pixels = 0
        black_pixels_x = 0
        black_pixels_y = 0

        white_pixels = 0
        white_pixels_x = 0
        white_pixels_y = 0

        height, width = eye_img.shape[:2]
        blank_image = eye_img.copy()
        blank_image2 = eye_img.copy()

        for y in range(height):
            for x in range(width):
                # Get RGB
                hsv = hsv_img[y][x]

                h = int(hsv[0])
                s = int(hsv[1])
                v = int(hsv[2])
                # bgr = eye_img[y][x]
                # r = int(bgr[2])
                # g = int(bgr[1])
                # b = int(bgr[0])
                # define range of blue color in HSV
                # lower_blue = np.array([110, 50, 50])
                # upper_blue = np.array([130, 255, 255])
                lower_blue = np.array([110, 50, 50])
                upper_blue = np.array([170, 255, 255])
                blank_image2 = cv2.inRange(hsv_img, lower_blue, upper_blue)

                if 110 < h < 130 and 50 < s < 255 and 50 < v < 255:
                    blank_image[y][x] = (0, 0, 255)

                    # Check if white
                    # min_white = (177, 142, 152)
                    # if r > min_white[0] and g > min_white[1] and b > min_white[2]:
                    #     if abs(height / 2 - y) < height / 2 * 0.5 \
                    #             and abs(width / 2 - x) < width / 2 * 0.75:
                    #         blank_image[y][x] = (0, 0, 255)
                    #         white_pixels += 1
                    #         white_pixels_x += x
                    #         white_pixels_y += y
                    #
                    # # Check if black
                    # max_black = (122, 82, 92)
                    # if r < max_black[0] and g < max_black[1] and b < max_black[2]:
                    #     if abs(height / 2 - y) < height / 2 * 0.3 \
                    #             and abs(width / 2 - x) < width / 2 * 0.30:
                    #         blank_image[x][y] = (0, 255, 0)
                    #         black_pixels += 1
                    #         black_pixels_x += x
                    #         black_pixels_y += y

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

        if black_pixels > 0:
            # check is black in the middle
            black_pixels_x = black_pixels_x / black_pixels
            black_pixels_y = black_pixels_y / black_pixels
            black_x_diff = abs(black_pixels_x - middle_x) / middle_x * 100
            black_y_diff = abs(black_pixels_y - middle_y) / middle_y * 100
            black_fill = black_pixels / n_pixels * 100
            if black_fill > 2:
                is_open = True
        return is_open, hsv_img, blank_image2

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

        if len(img1.shape) == 2:
            vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)

            vis[:h1, :w1] = img1
            vis[:h2, w1:w1 + w2] = img2
        else:
            vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

            vis[:h1, :w1, :3] = img1
            vis[:h2, w1:w1 + w2, :3] = img2

        return vis

    @staticmethod
    def vconcat(img1, img2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if len(img1.shape) == 2:
            vis = np.zeros((h1 + h2, max(w1, w2)), np.uint8)

            vis[:h1, :w1] = img1
            vis[h1:h1 + h2, :w2] = img2
        else:
            vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)

            vis[:h1, :w1, :3] = img1
            vis[h1:h1 + h2, :w2, :3] = img2

        return vis
