import cv2
import numpy as np


def display_text(img, text, position, margin=0):
    height, width = img.shape[:2]
    pos_x = position[0]
    pos_y = position[1]

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 0, 255)
    text_size = cv2.getTextSize(
        text=text,
        fontFace=font_face,
        fontScale=font_scale,
        thickness=thickness
    )
    if position[0] == -1:  # align left
        pos_x = margin
    elif position[0] == -2:  # align right
        pos_x = width - text_size[0][0] - margin
    if position[1] == -1:  # align top
        pos_y = text_size[0][1] + margin
    elif position[1] == -2:  # align bottom
        pos_y = height - margin

    position = (pos_x, pos_y)

    cv2.putText(img=img, text=text, org=position,
                fontFace=font_face, fontScale=font_scale,
                color=color, thickness=thickness)


def show_fps(img, fps):
    text = 'FPS: {0:.0f}'.format(fps)
    margin = 5
    position = (-1, -2)
    display_text(img=img, text=text, position=position, margin=margin)


def show_times(img, open_time, avg_time):
    text = 'open: {0:.2f}s'.format(open_time)
    margin = 5
    position = (-2, -1)
    display_text(img=img, text=text, position=position, margin=margin)

    text = 'avg: {0:.2f}s'.format(avg_time)
    position = (-2, 60)
    display_text(img=img, text=text, position=position, margin=margin)


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
