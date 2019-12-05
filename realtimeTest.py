# import cv2
# from collections import deque
# import numpy as np
# from keras.models  import load_model
#
# model = load_model("MNIST-CNN.model")
# lower_range = np.array([110, 50, 50])
# upper_range = np.array([130, 255, 255])
# cap = cv2.VideoCapture(0)
# point = deque(maxlen=132)
# while(cap.isOpened()):
#     ret, img = cap.read()
#     img = cv2.flip(img,1)
#     # img = cv2.rectangle(img,(600,500),(300,50),(255,255,255),1)
#     img = img[50:500,300:600]
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.inRange(hsv, lower_range, upper_range)
#     mask = cv2.erode(mask, kernel, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.dilate(mask, kernel, iterations=1)
#     res = cv2.bitwise_and(img, img, mask=mask)
#     cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#     center = None


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage
# import math
# import tensorflow as tf
# from keras.models import load_model
# from keras import backend as K
# from keras.models import load_model
#
# model = tf.keras.models.load_model('MNIST-CNN.model')
# img  = cv2.imread('digitImage.jpg',0)
# img_org  = cv2.imread('digitImage.jpg')
#
# rows,cols = img.shape
# img = cv2.GaussianBlur(img,(9,9),0)
# ret, img = cv2.threshold(img, 100, 255, 0)
# img = cv2.bitwise_not(img)
# contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# for j, cnt in enumerate(contours):
#     epsilon = 0.01 * cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     hull = cv2.convexHull(cnt)
#     k = cv2.isContourConvex(cnt)
#     x, y, w, h = cv2.boundingRect(cnt)
#     if (w > 1 and h > 50):
#         cv2.rectangle(img_org, (x, y-10), (x + w+10, y + h+10), (0, 255, 0), 2)
#         roi = img[y:y + h, x:x + w]
#         org_size = 28
#         img_size = 28
#         rows, cols = roi.shape
#         roi = img
#         if rows > cols:
#             factor = org_size / rows
#             rows = org_size
#             cols = int(round(cols * factor))
#         else:
#             factor = org_size / cols
#             cols = org_size
#             rows = int(round(rows * factor))
#             img = cv2.resize(roi, (28,28))
#             rows, cols = roi.shape[:2]
#         print(rows, cols)
#         img = cv2.resize(roi, (cols, rows))
#         colsPadding = ((abs(math.ceil((img_size - cols) / 2.0))),(abs(math.floor((img_size - cols) / 2.0))))
#         rowsPadding = (abs(math.ceil((img_size - rows) / 2.0)), abs(math.floor((img_size - rows) / 2.0)))
#         print(rowsPadding, colsPadding)
#         img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')
#         test_image = img.reshape(-1, 28, 28, 1)
#         pred = np.argmax(model.predict(test_image))
#         print(pred)
#         (x, y), radius = cv2.minEnclosingCircle(cnt)
#         def put_label(img, pred, x, y):
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             rect_x = int(x) + 10
#             rect_y = int(y) + 10
#             cv2.rectangle(img_org, (rect_x, rect_y + 5), (rect_x + 35, rect_y - 35), (0, 0, 255), cv2.FILLED)
#             cv2.putText(img_org, str(pred), (rect_x, rect_y), font, 1.5, (255, 0, 0), 1, cv2.LINE_AA)
#             return img_org
#         img_org = put_label(img, pred, x, y)
# cv2.imshow("image1", img_org)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import imutils
import pyautogui
from keras.models  import load_model
import numpy as np
import cv2
from PIL import Image
from mss import mss
import grab_screen
model = load_model("MNIST-CNN.model")

# image = cv2.imread("digitImage.jpg")
# image1 = cv2.imread("digitImage.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.GaussianBlur(image, (5, 5), 0)
cap = cv2.VideoCapture('video2.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640,480))
while(True):
    ret, image = cap.read()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.waitKey(27)
    ctrs, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,ctrs,-1,(255,255,0),2)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        x,y,w,h = rect
        if  h > 50 and h < 300  or w > 10 :
            cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            leng = int(rect[3] * 1.6)
            pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
            pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))
            roi = img[pt1:pt1+leng, pt2:pt2+leng]
            roi = cv2.resize(roi,(28, 28), interpolation=cv2.INTER_AREA)
            # roi = roi.reshape(1,784)
            roi = roi.reshape(-1,28, 28, 1)
            # roi=cv2.dilate(roi,(3,3))
            roi = np.array(roi, dtype='float32')
            roi /= 255
            pred_array = model.predict(roi)
            pred_array = np.argmax(pred_array)
            print('Result: {0}'.format(pred_array))
            cv2.putText(image, str(pred_array), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("Result",image)
    out.write(image)
    k = cv2.waitKey(27)
    if k==27:
        break
cv2.destroyAllWindows()
cap.release()
out.release()