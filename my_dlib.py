import numpy as np
import cv2
import dlib
import time
from scipy.spatial import distance as dist

from cv_utils import display_text, show_fps


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def show_winks(img, left, right, total):
    text = 'Left: {0}'.format(left)
    MARGIN = 5
    position = (-1, -1)
    display_text(img=img, text=text, position=position, margin=MARGIN)

    text = 'Right: {0}'.format(right)
    position = (-1, 60)
    display_text(img=img, text=text, position=position, margin=MARGIN)

    text = 'Total: {0}'.format(total)
    position = (-1, 100)
    display_text(img=img, text=text, position=position, margin=MARGIN)


def start_dlib(filename=None):
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

    FULL_POINTS = list(range(0, 68))
    FACE_POINTS = list(range(17, 68))

    JAWLINE_POINTS = list(range(0, 17))
    RIGHT_EYEBROW_POINTS = list(range(17, 22))
    LEFT_EYEBROW_POINTS = list(range(22, 27))
    NOSE_POINTS = list(range(27, 36))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_OUTLINE_POINTS = list(range(48, 61))
    MOUTH_INNER_POINTS = list(range(61, 68))

    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3

    COUNTER_LEFT = 0
    TOTAL_LEFT = 0

    COUNTER_RIGHT = 0
    TOTAL_RIGHT = 0

    TOTAL = 0
    EYE_DELAY_FRAMES = 10
    eye_delay = 0
    left_delay = False
    right_delay = False

    # Start time
    start_time = time.time()
    MAX_DELAY = 0.1  # displays the frame rate every 0.1 second
    counter = 0
    fps = 0
    frame_time = time.time()

    if filename is not None:
        video_capture = cv2.VideoCapture(filename)
        video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        print('Video fps: {0}:'.format(video_fps))
        is_real_time = False
    else:
        video_capture = cv2.VideoCapture(0)
        is_real_time = True

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    while True:
        # Block fps if file
        if not is_real_time and (time.time() - frame_time) < 1/video_fps:
            continue
        else:
            frame_time = time.time()

        ret, frame = video_capture.read()
        # Flip vertical
        frame = cv2.flip(frame, 1)

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 0)

            left_wink, right_wink = False, False
            for rect in rects:
                x = rect.left()
                y = rect.top()
                x1 = rect.right()
                y1 = rect.bottom()

                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])

                left_eye = landmarks[LEFT_EYE_POINTS]
                right_eye = landmarks[RIGHT_EYE_POINTS]

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)

                # cv2.putText(frame, "E.A.R. Left : {:.2f}".format(ear_left), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #             (0, 255, 255), 2)
                # cv2.putText(frame, "E.A.R. Right: {:.2f}".format(ear_right), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #             (0, 255, 255), 2)

                if ear_left < EYE_AR_THRESH:
                    COUNTER_LEFT += 1
                else:
                    if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                        left_wink = True
                        left_delay = True
                        eye_delay = EYE_DELAY_FRAMES
                        TOTAL_LEFT += 1
                        print("Left eye winked")
                    COUNTER_LEFT = 0

                if ear_right < EYE_AR_THRESH:
                    COUNTER_RIGHT += 1
                else:
                    if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
                        right_wink = True
                        right_delay = True
                        eye_delay = EYE_DELAY_FRAMES
                        TOTAL_RIGHT += 1
                        print("Right eye winked")
                    COUNTER_RIGHT = 0

            if left_wink and right_wink \
                    or left_wink and right_delay \
                    or right_wink and left_delay:
                TOTAL += 1
                left_delay = False
                right_delay = False

            show_winks(frame, TOTAL_LEFT, TOTAL_RIGHT, TOTAL)

            if eye_delay > 0:
                eye_delay -= 1
            else:
                left_delay = False
                right_delay = False

            # count fps
            counter += 1
            if (time.time() - start_time) > MAX_DELAY:
                fps = counter / (time.time() - start_time)
                counter = 0
                start_time = time.time()
            show_fps(frame, fps)

            cv2.imshow("Faces found", frame)

        k = 0xFF & cv2.waitKey(1)

        if k == ord('q') or k == 27:
            break

    cv2.destroyAllWindows()
