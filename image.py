import cv2


def show_image():
    image = cv2.imread("zajonc.jpg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Zajonc", image)
    cv2.imshow("Zajonc - odcienie szarości", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        # roi_gray = gray[x:x + w, y:y + h]
        roi_color = img[y:y + h, x:x + w]
        # roi_color = img[x:x + w, y:y + h]
        cv2.imwrite('face.jpg', roi_color)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        i = 0
        for (ex, ey, ew, eh) in eyes:
            ab_ey = y + ey
            ab_ex = x + ex
            eye_img = img[ab_ey:ab_ey + eh, ab_ex:ab_ex + ew]
            cv2.imshow('eye_img_' + str(i), eye_img)
            dir = 'eyes\\'
            cv2.imwrite(dir + str(i) + '_eye.jpg', eye_img)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            i += 1

    return img


def img2pixels(img):
    # Set threshold and maxValue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 80
    maxValue = 255

    # Basic threshold example
    th, dst = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY);

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    cv2.imshow('Normal', img)
    cv2.imshow('Gray', gray)
