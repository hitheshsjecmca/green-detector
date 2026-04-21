import cv2
import numpy as np

cap = cv2.VideoCapture(0)

background = cv2.imread("background.jpg")
background = cv2.resize(background, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([36, 44, 40])
    upper_green = np.array([84, 256, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    person = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background, background, mask=mask)

    final = cv2.add(person, bg)

    cv2.imshow("FINAL OUTPUT", final)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()