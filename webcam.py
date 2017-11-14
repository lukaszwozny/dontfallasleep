import cv2

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

    all_eyes_with_hight = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_color = img_copy[y:y + h, x:x + w]

        eyes = detect_eyes(gray_roi_img=roi_gray)
        i = 0
        for (ex, ey, ew, eh) in eyes:
            ab_ey = y + ey
            ab_ex = x + ex
            eye_img = img[ab_ey:ab_ey + eh, ab_ex:ab_ex + ew]
            # show_all_detected_eyes(eye_img, i, img_all_eyes)
            all_eyes_with_hight.append((eye_img, ey))
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            i += 1

    # if img_all_eyes is not None:
    #     cv2.imshow('eyes', img_all_eyes)
    # print(len(all_eyes))
    show_all_detected_eyes(all_eyes_with_hight)
    return img_copy


def show_all_detected_eyes(eyes):
    i = 0
    max_value = 0
    max_value_2 = 0
    for eye in eyes:
        y = eye[1]
        if y > max_value:
            max_value_2 = max_value
            max_value = y
        elif y > max_value_2:
            max_value_2 = y

    display_iter = 0
    for eye in eyes:
        y = eye[1]
        if y == max_value or y == max_value_2:
            cv2.imshow('test' + str(display_iter), eye[0])
            display_iter += 1
            # cv2.imwrite('eyes//eye_' + str(i) + '.jpg', eye)
        i += 1
