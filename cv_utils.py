import cv2


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