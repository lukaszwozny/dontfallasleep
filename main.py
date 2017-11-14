import cv2
import webcam

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img, 1)  # flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get all images
    img_detected = webcam.get_detected_img(img=img, gray_img=gray)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    # cv2.imshow('lap', sobelx)

    # connect all images
    org_det_img = cv2.hconcat((img, img_detected))
    org_det_img2 = cv2.hconcat((sobelx, sobely))
    # all_img = cv2.vconcat((org_det_img, org_det_img2))

    # Show all images
    cv2.imshow('img', org_det_img)
    # cv2.imshow('img2', org_det_img2)

    # Close when ESC pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
