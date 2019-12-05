import cv2
from collections import deque
import numpy as np
from keras.models  import load_model

model = load_model("MNIST-CNN.model")
lower_range = np.array([110, 50, 50])
upper_range = np.array([130, 255, 255])
cap = cv2.VideoCapture(0)
point = deque(maxlen=132)
while(cap.isOpened()):
    ret, img = cap.read()
    img = cv2.flip(img,1)
    # img = cv2.rectangle(img,(600,500),(300,50),(255,255,255),1)
    img = img[50:500,300:600]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    res = cv2.bitwise_and(img, img, mask=mask)
    cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 5:
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)

    point.appendleft(center)
    for i in range(1, len(point)):
        if point[i - 1] is None or point[i] is None:
            continue
        thick = int(np.sqrt(len(point) / float(i + 1)) * 2.5)
        cv2.line(img, point[i - 1], point[i], (0, 0, 225), thick)
        b = cv2.waitKey(10)
        # ************************************************************************************************************************
        if b == ord('b'):  # press 'b' to capture the background
            ctrs, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in ctrs]
            for rect in rects:
                print(rect)
                cv2.rectangle(mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
                leng = int(rect[3] * 1.6)
                print(leng)
                pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
                pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))
                roi = mask[pt1:pt1 + leng, pt2:pt2 + leng]
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                # roi = roi.reshape(1,784)
                roi = roi.reshape(-1, 28, 28, 1)
                # roi=cv2.dilate(roi,(3,3))
                roi = np.array(roi, dtype='float32')
                roi /= 255
                pred_array = model.predict(roi)
                pred_array = np.argmax(pred_array)
                print('Result: {0}'.format(pred_array))
                cv2.putText(img, str(pred_array), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)

    # *************************************************************************************************************************
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    cv2.imshow("Frame", img)

    k = cv2.waitKey(1)
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()


